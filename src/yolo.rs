// YOLO 目标检测模块
use anyhow::Result;
use std::sync::Arc;
use tract_onnx::prelude::*;

/// 检测结果
#[derive(Debug, Clone)]
pub struct Detection {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
    pub confidence: f32,
    pub class_id: i32,
}

/// YOLO 检测器
pub struct YoloDetector {
    model: Arc<TypedRunnableModel>,
    input_shape: (usize, usize), // (height, width)
}

/// Letterbox 几何参数：正向缩放/填充量，以及反变换回原图坐标所需的原图尺寸。
struct Letterbox {
    scale: f32,
    pad_x: f32,
    pad_y: f32,
    orig_w: f32,
    orig_h: f32,
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

    pub fn detect(&self, image_bytes: &[u8], conf_threshold: f32) -> Result<Vec<Detection>> {
        use image::ImageReader;
        use std::io::Cursor;

        // 解码图像
        let img = ImageReader::new(Cursor::new(image_bytes))
            .with_guessed_format()?
            .decode()?;

        let orig_w = img.width() as f32;
        let orig_h = img.height() as f32;
        let (target_h, target_w) = self.input_shape;
        let target_w = target_w as f32;
        let target_h = target_h as f32;

        // Letterbox 预处理（与 ultralytics 对齐）
        // 计算缩放比例，保持宽高比
        let scale = (target_w / orig_w).min(target_h / orig_h);
        let new_w = (orig_w * scale).round() as u32;
        let new_h = (orig_h * scale).round() as u32;

        // 等比缩放
        // 使用 Triangle (BILINEAR) 插值，与 ultralytics 的 cv2.resize 对齐
        let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();

        // 计算填充量（居中填充）
        let pad_x = ((target_w - new_w as f32) / 2.0).round() as u32;
        let pad_y = ((target_h - new_h as f32) / 2.0).round() as u32;

        // 创建 640×640 灰色画布并粘贴缩放图
        let tw = target_w as u32;
        let th = target_h as u32;
        let mut canvas = image::RgbImage::from_pixel(tw, th, image::Rgb([114u8, 114, 114]));
        image::imageops::overlay(&mut canvas, &rgb, pad_x as i64, pad_y as i64);

        // 转换为 NCHW float32 格式
        let mut input_data = Vec::with_capacity(3 * th as usize * tw as usize);
        for c in 0..3 {
            for y in 0..th as usize {
                for x in 0..tw as usize {
                    let pixel = canvas.get_pixel(x as u32, y as u32);
                    input_data.push(pixel[c] as f32 / 255.0);
                }
            }
        }

        let input_tensor: Tensor =
            tract_ndarray::Array4::from_shape_vec((1, 3, th as usize, tw as usize), input_data)?
                .into();

        // 推理
        let outputs = self.model.run(tvec!(input_tensor.into()))?;

        // 后处理：解析输出并反 letterbox 到原图坐标
        let letterbox = Letterbox {
            scale,
            pad_x: pad_x as f32,
            pad_y: pad_y as f32,
            orig_w,
            orig_h,
        };
        let detections = self.parse_outputs(&outputs, conf_threshold, &letterbox)?;

        Ok(detections)
    }

    fn parse_outputs(
        &self,
        outputs: &[tract_onnx::prelude::TValue],
        conf_threshold: f32,
        lb: &Letterbox,
    ) -> Result<Vec<Detection>> {
        let mut detections = Vec::new();

        if outputs.is_empty() {
            return Ok(detections);
        }

        let output = outputs[0].to_plain_array_view::<f32>()?;

        // YOLO26 端到端输出格式：[batch, num_detections, 6]
        // 每个检测：[x1, y1, x2, y2, confidence, class_id]（在 640×640 letterbox 空间）
        if output.ndim() >= 3 {
            let num_dets = output.shape()[1];
            for i in 0..num_dets {
                let conf = output[[0, i, 4]];
                if conf < conf_threshold {
                    continue;
                }

                let x1 = output[[0, i, 0]];
                let y1 = output[[0, i, 1]];
                let x2 = output[[0, i, 2]];
                let y2 = output[[0, i, 3]];
                let class_id = output[[0, i, 5]] as i32;

                // 反 letterbox：去除填充 → 除以缩放比 → 回到原图坐标
                let x_min = ((x1 - lb.pad_x) / lb.scale).max(0.0).min(lb.orig_w);
                let y_min = ((y1 - lb.pad_y) / lb.scale).max(0.0).min(lb.orig_h);
                let x_max = ((x2 - lb.pad_x) / lb.scale).max(0.0).min(lb.orig_w);
                let y_max = ((y2 - lb.pad_y) / lb.scale).max(0.0).min(lb.orig_h);

                detections.push(Detection {
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    confidence: conf,
                    class_id,
                });
            }
        }

        Ok(detections)
    }

    /// 对 class 0 框做 NMS
    pub fn nms(boxes: &mut Vec<Detection>, iou_threshold: f32) {
        if boxes.len() <= 1 {
            return;
        }

        // 按置信度降序排序
        boxes.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut kept = Vec::new();
        for box_ in boxes.iter() {
            let mut keep = true;
            for k in &kept {
                if iou(box_, k) > iou_threshold {
                    keep = false;
                    break;
                }
            }
            if keep {
                kept.push(box_.clone());
            }
        }

        *boxes = kept;
    }
}

fn iou(a: &Detection, b: &Detection) -> f32 {
    let xa = a.x_min.max(b.x_min);
    let ya = a.y_min.max(b.y_min);
    let xb = a.x_max.min(b.x_max);
    let yb = a.y_max.min(b.y_max);

    let inter_w = (xb - xa).max(0.0);
    let inter_h = (yb - ya).max(0.0);
    let inter_area = inter_w * inter_h;

    if inter_area == 0.0 {
        return 0.0;
    }

    let area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min);
    let area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min);

    inter_area / (area_a + area_b - inter_area)
}
