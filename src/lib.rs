//! gtlv 求解的纯 Rust functional core。
//!
//! 只放**核心推理逻辑 + 领域数据类型**：YOLO 检测、Siamese 特征提取、检测/裁剪/k-from-aspect
//! 编排。**无跨语言绑定、无网络 I/O、确定性**——各消费壳（wasm via wire FFI、pyo3 via PyO3）
//! 按自己的跨语言界面各写 wrapper，不在此复用边界层（见项目 core-shell 边界原则）。
//!
//! 目前产物为 [`types::DetectResult`]（检测框 + 特征），矩形指派/坐标输出待 Go 侧移植进来后
//! 由 core 直接提供 `solve_click`（M1b）。

pub mod engine;
pub mod perf;
pub mod siamese;
pub mod types;
pub mod yolo;

pub use engine::Engine;
pub use perf::PerfTimer;
pub use types::{BoundingBox, DetectResult, Detection, PerfTimings};
