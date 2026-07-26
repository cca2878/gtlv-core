//! Siamese 特征提取：裁剪 → 96×96 → 512 维嵌入。
//!
//! 预处理与推理分成两步导出，调用方可分别计时，也可复用已经预处理好的张量。

use anyhow::Result;
use std::sync::Arc;
use tract_onnx::prelude::*;

pub struct SiameseExtractor {
    model: Arc<TypedRunnableModel>,
    input_shape: (usize, usize), // (height, width)
}

impl SiameseExtractor {
    pub fn new(model_bytes: &[u8]) -> Result<Self> {
        // 内嵌固定为 NNEF（.nnef.tgz）；tract 从 reader 读 tgz（内部 GzDecoder + tar）。
        log::info!("Loading Siamese NNEF model from bytes");
        let model = tract_nnef::nnef()
            .model_for_read(&mut std::io::Cursor::new(model_bytes))?
            .with_input_fact(0, f32::fact([1, 3, 96, 96]))?
            .into_optimized()?
            .into_runnable()?;

        Ok(Self {
            model,
            input_shape: (96, 96),
        })
    }

    /// 裁剪指定像素区域并预处理成模型输入张量（裁剪 + 缩放 + 归一化）。
    pub fn preprocess_crop(
        &self,
        img: &image::DynamicImage,
        x_min: f32,
        y_min: f32,
        x_max: f32,
        y_max: f32,
    ) -> Result<Tensor> {
        let img_w = img.width() as f32;
        let img_h = img.height() as f32;

        let crop_x = x_min.clamp(0.0, img_w) as u32;
        let crop_y = y_min.clamp(0.0, img_h) as u32;
        let crop_w = ((x_max - x_min).max(1.0)).min(img_w - crop_x as f32) as u32;
        let crop_h = ((y_max - y_min).max(1.0)).min(img_h - crop_y as f32) as u32;

        if crop_w == 0 || crop_h == 0 {
            anyhow::bail!(
                "Degenerate crop region: {}x{} (img {}x{})",
                crop_w,
                crop_h,
                img.width(),
                img.height()
            );
        }

        self.preprocess(&img.crop_imm(crop_x, crop_y, crop_w, crop_h))
    }

    /// 缩放到 96×96 并转成 NCHW float32 张量。
    ///
    /// 插值用 Triangle(BILINEAR)，对齐训练时的 torchvision.transforms.Resize。
    pub fn preprocess(&self, img: &image::DynamicImage) -> Result<Tensor> {
        let (h, w) = self.input_shape;
        let rgb = img
            .resize_exact(w as u32, h as u32, image::imageops::FilterType::Triangle)
            .to_rgb8();

        let mut input_data = Vec::with_capacity(3 * h * w);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    input_data.push(rgb.get_pixel(x as u32, y as u32)[c] as f32 / 255.0);
                }
            }
        }

        Ok(tract_ndarray::Array4::from_shape_vec((1, 3, h, w), input_data)?.into())
    }

    /// 对预处理好的张量跑模型，返回特征向量。
    pub fn infer(&self, input: Tensor) -> Result<Vec<f32>> {
        let outputs = self.model.run(tvec!(input.into()))?;
        if outputs.is_empty() {
            anyhow::bail!("Siamese model returned empty output");
        }
        Ok(outputs[0]
            .to_plain_array_view::<f32>()?
            .iter()
            .copied()
            .collect())
    }
}
