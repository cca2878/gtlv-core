// Siamese 特征提取模块
use anyhow::Result;
use std::sync::Arc;
use tract_onnx::prelude::*;

/// 特征提取器
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

    pub fn extract_features_from_image(&self, img: &image::DynamicImage) -> Result<Vec<f32>> {
        let (h, w) = self.input_shape;

        // Resize 到 96x96
        // 使用 Triangle (BILINEAR) 插值，与训练时的 torchvision.transforms.Resize 对齐
        let resized = img.resize_exact(w as u32, h as u32, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();

        // 转换为 NCHW float32 格式（归一化到 [0, 1]）
        let mut input_data = Vec::with_capacity(3 * h * w);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let pixel = rgb.get_pixel(x as u32, y as u32);
                    input_data.push(pixel[c] as f32 / 255.0);
                }
            }
        }

        let input_tensor: Tensor =
            tract_ndarray::Array4::from_shape_vec((1, 3, h, w), input_data)?.into();

        // 推理
        let outputs = self.model.run(tvec!(input_tensor.into()))?;

        // 提取特征向量
        if outputs.is_empty() {
            anyhow::bail!("Siamese model returned empty output");
        }

        let output = outputs[0].to_plain_array_view::<f32>()?;
        let features: Vec<f32> = output.iter().copied().collect();

        Ok(features)
    }

    /// 从图像裁剪区域提取特征（像素坐标）
    pub fn extract_features_from_crop(
        &self,
        img: &image::DynamicImage,
        x_min: f32,
        y_min: f32,
        x_max: f32,
        y_max: f32,
    ) -> Result<Vec<f32>> {
        let img_w = img.width() as f32;
        let img_h = img.height() as f32;

        // 像素坐标裁剪（已在调用侧转换为原图像素坐标）
        let crop_x = (x_min).max(0.0).min(img_w) as u32;
        let crop_y = (y_min).max(0.0).min(img_h) as u32;
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

        let cropped = img.crop_imm(crop_x, crop_y, crop_w, crop_h);
        self.extract_features_from_image(&cropped)
    }
}
