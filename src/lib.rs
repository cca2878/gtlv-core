//! gtlv 求解的纯 Rust functional core。
//!
//! 只放**核心推理逻辑 + 领域数据类型**：YOLO 检测、Siamese 特征提取、检测/裁剪/k-from-aspect
//! 编排。**无跨语言绑定、无网络 I/O、确定性**——各消费壳（wasm via wire FFI、pyo3 via PyO3）
//! 按自己的跨语言界面各写 wrapper，不在此复用边界层（见项目 core-shell 边界原则）。
//!
//! 产物是 [`types::DetectResult`]（检测框 + 特征 + k）。**把提示词指派到答案格属于纯数学后处理，
//! 刻意不放在这里**——它不依赖推理引擎，各壳按自己语言的惯用法自行实现，core 只提供算法契约
//! `docs/matching.md`（含必须用 f64 累加距离等精度约定）。

pub mod engine;
pub mod perf;
pub mod siamese;
pub mod types;
pub mod yolo;

pub use engine::Engine;
pub use perf::PerfTimer;
pub use types::{BoundingBox, DetectResult, Detection, PerfTimings};
