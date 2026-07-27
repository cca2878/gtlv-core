# gtlv-core

极验（GeeTest）点选验证码的推理核心，以纯 Rust 实现。

对于一张验证码图像，本 crate 输出答案格的位置、各答案格与提示词各字的特征向量，以及提示词字数 k。特征到点击坐标的转换由调用方完成，其规格见[匹配契约](#匹配契约)一节。

- **检测**：YOLO（nano ONNX），单次前向同时给出提示框与答案格
- **特征提取**：Siamese（NNEF），96×96 输入，输出 512 维嵌入
- **字数标定**：依据提示框长宽比得出 k，不依赖检测到的答案格数量

本 crate 不含网络 I/O 与跨语言绑定，行为确定。推理后端采用[tract](https://github.com/sonos/tract)，为纯 Rust 实现，不依赖 ONNX Runtime 或 CGO，可编译至 `wasm32-wasip1` 及各原生 target。

## 用法

模型已内置，构造完成后即可求解。模型加载与图优化需数百毫秒，因此实例应在进程内复用。

```rust
use gtlv_core::{Engine, PerfTimer};

let engine = Engine::new()?;                     // 加载内置模型，每进程执行一次
let timer = PerfTimer::new(false);               // 传入 true 可获得分阶段耗时

let result = engine.detect_and_extract(&image_bytes, 0.5, &timer)?;

let m = result.detections.len();                 // 答案格数量，含干扰字
let k = result.prompt_features.len();            // 提示词字数
for (detection, features) in result.detections.iter().zip(&result.answer_features) {
    // detection.bbox 为原图像素坐标，features 为该答案格的 512 维嵌入
}
```

`detections` 与 `answer_features` 一一对应且顺序一致；`prompt_features` 按提示词自左至右排列。若 `prompt_box` 为 `None`、k 不在 2 至 4 之间，或 m 小于 k，则表明该图像无法求解，应更换图像重试。

依赖声明：

```toml
gtlv-core = { git = "https://github.com/cca2878/gtlv-core", rev = "..." }
```

由于模型体积超出 crates.io 的单包上限，本 crate 仅通过 git 依赖分发。

## 模型内置

模型经 `include_bytes!` 编入 crate，依赖本 crate 即自动获得，无需外部模型文件、文件系统挂载或临时目录，这一特性对 WASM 与移动端部署尤为重要。相应地，产物将包含约 26MB 模型数据（编译所得的 wasm 模块约为 28MB）。

模型权重已经过压缩，再行无损压缩收效甚微，故以原始形式内置。更换模型的方式为替换 `modeldata/`下的文件并重新编译调用方产物。

## 匹配契约

将提示词指派至答案格的实现不在本 crate 内。该步骤为纯数学计算——欧氏距离与矩形最优指派——既不依赖推理引擎，也不涉及跨语言边界，宜由各调用方以其语言的惯用方式实现。

算法规格见 [`docs/matching.md`](docs/matching.md)，其中包含前置校验、指派约束，以及影响结果正确性的精度要求（距离须以 f64 累加）。遵循该规格的不同语言实现将产生一致的点击顺序。

## 构建

Rust 版本及所需 target 由 `rust-toolchain.toml` 声明，rustup 将自动安装。

```bash
cargo test
cargo build --release --target wasm32-wasip1   # 供 WASM 调用方使用
```

## 许可

[AGPL-3.0-only](./LICENSE)，与 gtlv 系其余项目及上游 Amorter/biliTicker_gt 保持一致。
