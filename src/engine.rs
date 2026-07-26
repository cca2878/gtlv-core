//! 推理编排：解码 → YOLO 检测 → NMS/Top-K 过滤 → 裁剪 + Siamese 特征提取。
//!
//! 纯逻辑，产物为 [`crate::types::DetectResult`]（检测框 + 特征）；序列化由各壳自行 wrapper。
use anyhow::Result;

use crate::perf::PerfTimer;
use crate::siamese::SiameseExtractor;
use crate::types::{BoundingBox, DetectResult, Detection, PerfTimings};
use crate::yolo::YoloDetector;

/// 答案格数量上限，用于限制检测异常时的框数。
///
/// 取值显著大于提示词字数：答案网格含有干扰字，格数通常多于字数。
/// 重复框由 NMS 消除，应点击的格子由调用方的指派确定。
const MAX_ANS: usize = 8;

/// 由提示框长宽比标定提示词字数 k。
///
/// k 不得由检测到的答案格数量推断：答案网格含有不应点击的干扰字，格数通常多于字数
/// （常见多一个），据此推断将导致点击错误。阈值取自真机样本，在 501 张图像上判定全部正确。
pub fn prompt_char_count(width: f32, height: f32) -> usize {
    let aspect = width / height.max(1.0);
    if aspect < 1.95 {
        2
    } else if aspect < 2.60 {
        3
    } else {
        4
    }
}

/// 承载已加载模型的推理引擎（一次加载，多次求解；无内部可变状态，可并发调用）。
pub struct Engine {
    yolo: YoloDetector,
    siamese: SiameseExtractor,
}

impl Engine {
    /// nano yolo 输入尺寸（选定规格 nano@384）。
    const YOLO_INPUT: usize = 384;

    /// 内嵌生产模型（随 core 编入产物；wasm/native 各壳自动携带，无需外部文件/FS 挂载）。
    const YOLO_MODEL: &'static [u8] = include_bytes!("../modeldata/yolo26n_gt_v2_384.onnx");
    const SIAMESE_MODEL: &'static [u8] = include_bytes!("../modeldata/siamese_feature.nnef.tgz");

    pub fn new() -> Result<Self> {
        // 顺序加载：WASM guest 为单线程，无法借助 thread::scope 并行加载。
        let yolo = YoloDetector::new(Self::YOLO_MODEL, Self::YOLO_INPUT)?;
        let siamese = SiameseExtractor::new(Self::SIAMESE_MODEL)?;
        Ok(Self { yolo, siamese })
    }

    /// 检测 + 特征提取。`timer` 由调用方提供（不需要计时就传禁用的计时器）。
    pub fn detect_and_extract(
        &self,
        image_bytes: &[u8],
        conf_threshold: f32,
        timer: &PerfTimer,
    ) -> Result<DetectResult> {
        let total_start = std::time::Instant::now();

        // 1. 解码一次，检测与后续裁剪共用同一份像素。
        timer.start("preprocess");
        let img = image::load_from_memory(image_bytes)?;
        timer.stop("preprocess");

        timer.start("yolo_infer");
        let conf_threshold = if conf_threshold > 0.0 {
            conf_threshold
        } else {
            0.5
        };
        let detections = self.yolo.detect(&img, conf_threshold)?;
        timer.stop("yolo_infer");

        // 2. class 0 后处理：NMS 去重 + Top-K 兜底（与 Python filter_ans_boxes 对齐）
        let mut ans_boxes: Vec<_> = detections
            .iter()
            .filter(|d| d.class_id == 0)
            .cloned()
            .collect();
        YoloDetector::nms(&mut ans_boxes, 0.5);
        if ans_boxes.len() > MAX_ANS {
            // total_cmp 对 NaN 也是全序，置信度异常时不会 panic。
            ans_boxes.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
            ans_boxes.truncate(MAX_ANS);
        }

        // class 1：取置信度最高的提示词整体框
        let prompt_box = detections
            .iter()
            .filter(|d| d.class_id == 1)
            .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
            .cloned();

        let k = prompt_box.as_ref().map_or(0usize, |pb| {
            prompt_char_count(pb.x_max - pb.x_min, pb.y_max - pb.y_min)
        });

        // 3. 裁剪 + 特征提取，两个阶段各自计时。
        let mut out_detections: Vec<Detection> = Vec::new();
        let mut answer_features: Vec<Vec<f32>> = Vec::new();
        let mut prompt_features: Vec<Vec<f32>> = Vec::new();

        for (di, det) in ans_boxes.iter().enumerate() {
            match self.extract(&img, det.x_min, det.y_min, det.x_max, det.y_max, timer) {
                Ok(features) => {
                    out_detections.push(Detection {
                        bbox: BoundingBox {
                            x_min: det.x_min,
                            y_min: det.y_min,
                            x_max: det.x_max,
                            y_max: det.y_max,
                        },
                        confidence: det.confidence,
                        class_id: det.class_id,
                    });
                    answer_features.push(features);
                }
                Err(e) => {
                    log::warn!("Feature extraction failed for ans box {di}: {e}");
                    continue;
                }
            }
        }

        let out_prompt_box = prompt_box.as_ref().map(|pb| {
            // 按标定出的字数 k 等分提示框。
            let seg_width = (pb.x_max - pb.x_min) / (k.max(1)) as f32;
            for i in 0..k {
                let x_min = pb.x_min + i as f32 * seg_width;
                match self.extract(&img, x_min, pb.y_min, x_min + seg_width, pb.y_max, timer) {
                    Ok(features) => prompt_features.push(features),
                    Err(e) => {
                        log::warn!("Feature extraction failed for prompt segment {i}: {e}");
                        // 全或无：任一段失败即弃掉全部，交上层换图重试。
                        // 少返一段会让下游把 k 低估成实际段数，比整体失败更危险。
                        prompt_features.clear();
                        break;
                    }
                }
            }
            BoundingBox {
                x_min: pb.x_min,
                y_min: pb.y_min,
                x_max: pb.x_max,
                y_max: pb.y_max,
            }
        });

        Ok(DetectResult {
            detections: out_detections,
            answer_features,
            prompt_features,
            prompt_box: out_prompt_box,
            perf: PerfTimings {
                preprocess_ms: timer.elapsed_ms("preprocess"),
                yolo_infer_ms: timer.elapsed_ms("yolo_infer"),
                crop_resize_ms: timer.elapsed_ms("crop_resize"),
                siamese_infer_ms: timer.elapsed_ms("siamese_infer"),
                total_ms: total_start.elapsed().as_millis() as i64,
            },
        })
    }

    /// 裁剪指定区域并提取特征，裁剪与推理分别累加计时。
    fn extract(
        &self,
        img: &image::DynamicImage,
        x_min: f32,
        y_min: f32,
        x_max: f32,
        y_max: f32,
        timer: &PerfTimer,
    ) -> Result<Vec<f32>> {
        timer.start("crop_resize");
        let input = self
            .siamese
            .preprocess_crop(img, x_min, y_min, x_max, y_max);
        timer.stop("crop_resize");

        timer.start("siamese_infer");
        let features = self.siamese.infer(input?);
        timer.stop("siamese_infer");

        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_char_count_covers_each_threshold_band() {
        // 阈值 1.95 / 2.60，提示条高度固定 40px 左右。
        assert_eq!(prompt_char_count(70.0, 40.0), 2); // aspect 1.75
        assert_eq!(prompt_char_count(90.0, 40.0), 3); // aspect 2.25
        assert_eq!(prompt_char_count(120.0, 40.0), 4); // aspect 3.00
    }

    #[test]
    fn prompt_char_count_is_exact_at_the_boundaries() {
        // 边界归属：< 1.95 才是 2，恰好 1.95 归 3；< 2.60 才是 3，恰好 2.60 归 4。
        assert_eq!(prompt_char_count(1.95 * 40.0, 40.0), 3);
        assert_eq!(prompt_char_count(1.94 * 40.0, 40.0), 2);
        assert_eq!(prompt_char_count(2.60 * 40.0, 40.0), 4);
        assert_eq!(prompt_char_count(2.59 * 40.0, 40.0), 3);
    }

    #[test]
    fn prompt_char_count_survives_degenerate_height() {
        // 高度为 0 不能除爆；钳到 1.0 后按宽度落进某个档位即可。
        assert_eq!(prompt_char_count(1.0, 0.0), 2);
        assert_eq!(prompt_char_count(100.0, 0.0), 4);
    }
}
