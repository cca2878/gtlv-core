# gtlv-core

极验（GeeTest）点选验证码求解的**纯 Rust functional core**。

只放**核心推理逻辑 + 领域数据类型**：

- **YOLO 检测**（nano ONNX，同一次前向出提示框 class 1 + 答案格 class 0）
- **Siamese 特征提取**（NNEF，96² 输入、512-d 嵌入）
- **编排**：NMS/Top-K 过滤、k-from-aspect（由提示框长宽比标定字数 k）、裁剪 + 特征提取

**无跨语言绑定、无网络 I/O、确定性**。各消费壳按自己的跨语言界面各写 wrapper，不在此复用边界层
（functional-core / imperative-shell）：

- **wasm 壳**（`gtlv-go` 的 `rust-wasm`）：经 wazero 的字节 FFI（`wire` 序列化）供 Go 侧加载。
- **pyo3 壳**（[`gtlv-py`](https://github.com/cca2878/gtlv-py)）：经 PyO3 直接把 core 类型转 Python 对象。

推理后端为 [tract](https://github.com/sonos/tract)（纯 Rust、零 onnxruntime/CGO），可编 `wasm32-wasip1`
或各原生 target。

## 模型内嵌：core 是唯一来源

生产模型放 `modeldata/`，经 `include_bytes!` **编入 core**，`Engine::new()` 从内存字节加载
（tract `model_for_read`）。因此：

- **各语言壳不各自维护模型**——依赖 core 即自动携带，无需外部文件、无需 FS 挂载、**无临时目录依赖**。
- 换模型 = 换 `modeldata/` 后重编消费方产物（wasm 壳即 `make build-wasm`）。
- 代价：产物含模型（wasm 模块约 28MB）。实测 wazero 加载 28MB 模块的冷/暖启与 2MB 版**几乎相同**
  （6.4s / 2.1s）——模型是 data 段、非代码，AOT 编译量不变。
- 无损压缩对训练后权重**无效**（实测 xz 相对 gzip 仅省 0.8%），故原样内嵌、不额外压缩。

## 边界：core 只做推理，纯数学后处理归各壳

core 的产物是**检测框 + 特征向量 + k**（见 [`types::DetectResult`]）。把提示词特征指派到答案框的
**矩形最优指派**是纯数学后处理（欧氏距离 + 指派），不依赖推理引擎、不涉及跨语言边界，故
**各壳按自己语言的惯用法各自实现**，core 只提供算法契约文档：见 [`docs/matching.md`](docs/matching.md)。

## 复用方式

开发期消费方用 **path 依赖**（相对路径）引入；core 稳定后可切 **git rev 依赖**锁版本。

```toml
# 消费方 Cargo.toml
gtlv-core = { path = "../../gtlv-core" }              # 开发期
# gtlv-core = { git = "https://github.com/cca2878/gtlv-core", rev = "..." }  # 锁版本
```

## 许可

AGPL-3.0-only（与上游 Amorter/biliTicker_gt 及 gtlv 系一致）。
