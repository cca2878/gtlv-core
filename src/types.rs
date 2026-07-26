//! 领域数据类型：推理引擎的产物模型（纯 Rust，无序列化/无跨语言绑定）。
//!
//! 曾内联在 wasm 壳的 `wire` 模块里；抽入 core 后，wasm 壳的字节序列化与将来 pyo3 壳的
//! Python 对象转换都各自在壳内针对这些类型写 wrapper（边界层不复用，见 core-shell 原则）。

#[derive(Clone, Copy, Default)]
pub struct BoundingBox {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

#[derive(Clone, Copy)]
pub struct Detection {
    pub bbox: BoundingBox,
    pub confidence: f32,
    pub class_id: i32,
}

#[derive(Clone, Copy, Default)]
pub struct PerfTimings {
    pub preprocess_ms: i64,
    pub yolo_infer_ms: i64,
    pub crop_resize_ms: i64,
    pub siamese_infer_ms: i64,
    pub total_ms: i64,
}

/// 推理结果（引擎产物）。answer_features 与 detections 一一对应；prompt_features 按位置排列。
#[derive(Default)]
pub struct DetectResult {
    pub detections: Vec<Detection>,
    pub prompt_box: Option<BoundingBox>,
    pub answer_features: Vec<Vec<f32>>,
    pub prompt_features: Vec<Vec<f32>>,
    pub perf: PerfTimings,
}
