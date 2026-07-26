//! YOLO 目标检测：letterbox 预处理 → 推理 → 反变换回原图坐标 → NMS。

use anyhow::Result;
use std::sync::Arc;
use tract_onnx::prelude::*;

/// YOLO 的原始检测框（扁平坐标）。对外的聚合形态见 [`crate::types::Detection`]。
#[derive(Debug, Clone)]
pub struct RawDetection {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
    pub confidence: f32,
    pub class_id: i32,
}

pub struct YoloDetector {
    model: Arc<TypedRunnableModel>,
    input_shape: (usize, usize), // (height, width)
}

/// Letterbox 几何参数：正向缩放/填充量，以及反变换回原图所需的原图尺寸。
struct Letterbox {
    scale: f32,
    pad_x: f32,
    pad_y: f32,
    orig_w: f32,
    orig_h: f32,
}

impl Letterbox {
    /// 把 letterbox 空间的坐标反算回原图像素坐标，并钳制在图内。
    fn to_original(&self, x: f32, y: f32) -> (f32, f32) {
        (
            ((x - self.pad_x) / self.scale).clamp(0.0, self.orig_w),
            ((y - self.pad_y) / self.scale).clamp(0.0, self.orig_h),
        )
    }
}

impl YoloDetector {
    pub fn new(model_bytes: &[u8], input_size: usize) -> Result<Self> {
        log::info!("Loading YOLO model from bytes ({input_size}x{input_size})");

        let model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(model_bytes))?
            .with_input_fact(0, f32::fact([1, 3, input_size, input_size]).into())?
            .into_optimized()?
            .into_runnable()?;

        Ok(Self {
            model,
            input_shape: (input_size, input_size),
        })
    }

    /// 检测。接收**已解码**的图像——同一张图在一次求解里还要用于裁剪，解码只做一次。
    pub fn detect(
        &self,
        img: &image::DynamicImage,
        conf_threshold: f32,
    ) -> Result<Vec<RawDetection>> {
        let orig_w = img.width() as f32;
        let orig_h = img.height() as f32;
        let (target_h, target_w) = self.input_shape;
        let target_w = target_w as f32;
        let target_h = target_h as f32;

        // Letterbox 预处理（与 ultralytics 对齐）：等比缩放 + 居中灰边填充。
        // 插值用 Triangle(BILINEAR)，对齐 ultralytics 的 cv2.resize。
        let scale = (target_w / orig_w).min(target_h / orig_h);
        let new_w = (orig_w * scale).round() as u32;
        let new_h = (orig_h * scale).round() as u32;
        let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();

        let pad_x = ((target_w - new_w as f32) / 2.0).round() as u32;
        let pad_y = ((target_h - new_h as f32) / 2.0).round() as u32;

        let tw = target_w as u32;
        let th = target_h as u32;
        let mut canvas = image::RgbImage::from_pixel(tw, th, image::Rgb([114u8, 114, 114]));
        image::imageops::overlay(&mut canvas, &rgb, pad_x as i64, pad_y as i64);

        // 转 NCHW float32，归一化到 [0, 1]。
        let mut input_data = Vec::with_capacity(3 * th as usize * tw as usize);
        for c in 0..3 {
            for y in 0..th {
                for x in 0..tw {
                    input_data.push(canvas.get_pixel(x, y)[c] as f32 / 255.0);
                }
            }
        }
        let input_tensor: Tensor =
            tract_ndarray::Array4::from_shape_vec((1, 3, th as usize, tw as usize), input_data)?
                .into();

        let outputs = self.model.run(tvec!(input_tensor.into()))?;

        let letterbox = Letterbox {
            scale,
            pad_x: pad_x as f32,
            pad_y: pad_y as f32,
            orig_w,
            orig_h,
        };
        self.parse_outputs(&outputs, conf_threshold, &letterbox)
    }

    fn parse_outputs(
        &self,
        outputs: &[tract_onnx::prelude::TValue],
        conf_threshold: f32,
        lb: &Letterbox,
    ) -> Result<Vec<RawDetection>> {
        let mut detections = Vec::new();

        if outputs.is_empty() {
            return Ok(detections);
        }

        let output = outputs[0].to_plain_array_view::<f32>()?;

        // YOLO26 端到端输出：[batch, num_detections, 6]
        // 每项 [x1, y1, x2, y2, confidence, class_id]（letterbox 空间）。
        if output.ndim() >= 3 {
            let num_dets = output.shape()[1];
            for i in 0..num_dets {
                let conf = output[[0, i, 4]];
                // 必须显式排除非有限值：`NaN < threshold` 恒为 false，单靠阈值比较会让
                // NaN 混进结果，再在后续按置信度排序时炸掉。
                if !conf.is_finite() || conf < conf_threshold {
                    continue;
                }

                let (x_min, y_min) = lb.to_original(output[[0, i, 0]], output[[0, i, 1]]);
                let (x_max, y_max) = lb.to_original(output[[0, i, 2]], output[[0, i, 3]]);

                detections.push(RawDetection {
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    confidence: conf,
                    class_id: output[[0, i, 5]] as i32,
                });
            }
        }

        Ok(detections)
    }

    /// 按置信度降序做 NMS，抑制 IoU 超阈值的重复框。
    pub fn nms(boxes: &mut Vec<RawDetection>, iou_threshold: f32) {
        if boxes.len() <= 1 {
            return;
        }

        // total_cmp 对 NaN 也是全序，置信度异常时不会 panic。
        boxes.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));

        let mut kept: Vec<RawDetection> = Vec::new();
        for candidate in boxes.iter() {
            if kept.iter().all(|k| iou(candidate, k) <= iou_threshold) {
                kept.push(candidate.clone());
            }
        }

        *boxes = kept;
    }
}

fn iou(a: &RawDetection, b: &RawDetection) -> f32 {
    let inter_w = (a.x_max.min(b.x_max) - a.x_min.max(b.x_min)).max(0.0);
    let inter_h = (a.y_max.min(b.y_max) - a.y_min.max(b.y_min)).max(0.0);
    let inter_area = inter_w * inter_h;

    if inter_area == 0.0 {
        return 0.0;
    }

    let area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min);
    let area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min);

    inter_area / (area_a + area_b - inter_area)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x_min: f32, y_min: f32, x_max: f32, y_max: f32, confidence: f32) -> RawDetection {
        RawDetection {
            x_min,
            y_min,
            x_max,
            y_max,
            confidence,
            class_id: 0,
        }
    }

    #[test]
    fn iou_is_zero_when_disjoint_and_one_when_identical() {
        let a = det(0.0, 0.0, 10.0, 10.0, 1.0);
        assert_eq!(iou(&a, &det(20.0, 20.0, 30.0, 30.0, 1.0)), 0.0);
        assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_matches_hand_computed_overlap() {
        // 交 = 5x10 = 50，并 = 100 + 100 - 50 = 150。
        let got = iou(
            &det(0.0, 0.0, 10.0, 10.0, 1.0),
            &det(5.0, 0.0, 15.0, 10.0, 1.0),
        );
        assert!((got - 50.0 / 150.0).abs() < 1e-6, "got {got}");
    }

    #[test]
    fn nms_keeps_highest_confidence_and_drops_overlaps() {
        let mut boxes = vec![
            det(0.0, 0.0, 10.0, 10.0, 0.6),
            det(0.5, 0.5, 10.5, 10.5, 0.9), // 与上面高度重叠，置信度更高
            det(50.0, 50.0, 60.0, 60.0, 0.7), // 不重叠，应保留
        ];
        YoloDetector::nms(&mut boxes, 0.5);
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0].confidence, 0.9); // 按置信度降序
        assert_eq!(boxes[1].confidence, 0.7);
    }

    #[test]
    fn nms_does_not_panic_on_nan_confidence() {
        // 置信度排序必须对 NaN 全序，否则模型给出异常分值时会 panic。
        let mut boxes = vec![
            det(0.0, 0.0, 10.0, 10.0, f32::NAN),
            det(50.0, 50.0, 60.0, 60.0, 0.7),
        ];
        YoloDetector::nms(&mut boxes, 0.5);
        assert_eq!(boxes.len(), 2);
    }

    #[test]
    fn letterbox_inverts_padding_and_scaling() {
        // 原图 300x150 → 384x384：scale=1.28，缩放后 384x192，上下各留 96 像素。
        let lb = Letterbox {
            scale: 1.28,
            pad_x: 0.0,
            pad_y: 96.0,
            orig_w: 300.0,
            orig_h: 150.0,
        };
        let (x, y) = lb.to_original(128.0, 96.0);
        assert!((x - 100.0).abs() < 1e-3, "x={x}");
        assert!(y.abs() < 1e-3, "y={y}"); // 填充起点映射回原图顶边
    }

    #[test]
    fn letterbox_clamps_outside_original_bounds() {
        let lb = Letterbox {
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
            orig_w: 100.0,
            orig_h: 50.0,
        };
        assert_eq!(lb.to_original(-30.0, 999.0), (0.0, 50.0));
    }
}
