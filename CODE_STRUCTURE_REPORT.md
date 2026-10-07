# CUA 训练仓库结构说明

## 结论

仓库分为 `atomic/` 与 `complex/` 两个可以**单独取走**的训练版本。每个目录都有自己的启动脚本、数据下载和校验脚本、CUA 图文适配器、权重格式导入工具、依赖文件、测试和 README。两版之间没有 Python 导入，也不通过仓库根目录的代码运行。根 README 只用于导航。

“单独”指这两个版本在**本仓库内部**互不依赖。它不表示仓库包含数 GB 模型权重或 NVIDIA Megatron-Bridge 训练引擎：使用者仍需单独准备兼容的 CUDA/PyTorch、Megatron-Bridge 和初始化 checkpoint。

```text
cua_train/
├── README.md                         两版导航与公共环境说明
├── CODE_STRUCTURE_REPORT.md          本报告
├── atomic/
│   ├── README.md                     3x-full 版本说明
│   ├── requirements.txt              下载器依赖
│   ├── train.sh                      八卡训练入口
│   ├── run_qwen35_cua_1ep.py         Megatron-Bridge 训练入口
│   ├── download_data.py              HF atomic 数据下载与哈希校验
│   ├── cua_energon_adapter.py        CUA 图文格式适配
│   ├── import_qwen35_vlm_checkpoint.py  可选权重导入工具
│   └── test_download_data.py         本版数据校验测试
└── complex/
    ├── README.md                     Slides01 v4 版本说明
    ├── requirements.txt              下载器依赖
    ├── train.sh                      八卡训练入口
    ├── run_slides01_sft.py           Megatron-Bridge 训练入口
    ├── download_data.py              HF complex 数据下载与哈希校验
    ├── cua_energon_adapter.py        本版自己的 CUA 图文格式适配
    ├── import_qwen35_vlm_checkpoint.py  可选权重导入工具
    ├── action_window_provider.py     action 样本读取和确定性采样
    ├── action_window_encoder.py      只监督目标 action 的 loss mask
    └── test_download_data.py         本版数据校验测试
```

## 文件之间怎样调用

### Atomic

```text
atomic/train.sh func|gui
  ├─ atomic/download_data.py
  │    └─ HF: Furunhao/cua/dataset/cua-training-data/atomic/
  └─ atomic/run_qwen35_cua_1ep.py
       ├─ atomic/cua_energon_adapter.py
       └─ 外部 Megatron-Bridge + Megatron-LM
```

`train.sh` 下载数据，校验两份 JSONL 的 SHA256 和 `.jsonl.idx` 是否存在，设置图片根目录，再启动八卡训练。`cua_energon_adapter.py` 将 JSONL 的 `system/conversations/images` 转成 Qwen 图文编码器所需输入，处理消息角色，核对图片占位符，并从下载目录加载 JPG。

Function 与 GUI 是两条独立路线：数据分别为 53,917 与 53,918 条源轨迹；各训练一轮、422 次 optimizer 更新、global batch 128；Function 使用 CP2，GUI 使用 CP4。最终保存点是 `iter_0000422`。两路分别从 Qwen3.5-4B base 导入的 Megatron 权重开始，不互相续训。

### Complex

```text
complex/train.sh func|gui
  ├─ complex/download_data.py
  │    └─ HF: Furunhao/cua/dataset/cua-training-data/complex/
  └─ complex/run_slides01_sft.py
       ├─ complex/action_window_provider.py
       ├─ complex/action_window_encoder.py
       │    └─ complex/cua_energon_adapter.py
       └─ 外部 Megatron-Bridge + Megatron-LM
```

`complex` 下载 `action_window_current/` 的训练 JSONL、索引和转换 manifest，以及 `sft_gui_sharegpt/` 图片。HF 上的 `action_window_current` 是发布后的**普通目录**，对应原 DGX 上的 v4 符号链接目标。下载器以 manifest 的 SHA256 校验两份 JSONL。

`action_window_provider.py` 使用 `shuffle_buffer_size=0` 的确定性读取；`action_window_encoder.py` 在 CUA 适配器基础上检查样本里的目标 action 哈希与图片数量，并将历史 action 的 loss 屏蔽，只监督最后的目标 action 和 EOS。零 loss padding 样本不参与 loss。

Function 每 epoch 有 47,163 条真实 action + 5 条 padding、737 次更新、global batch 64；GUI 有 206,406 条真实 action + 26 条 padding、6,451 次更新、global batch 32。两路均跑三个 epoch、CP4，最多 7 张图，每 epoch 保存一个完整 checkpoint。

Complex 的初始权重必须分别来自**相同路线**的 atomic `iter_0000422`。它继承模型参数，重新开始 optimizer、scheduler、RNG 和训练步数。GitHub 与 HF 数据集中没有附带这个初始化权重。

## 为什么两个目录都有同名文件

`download_data.py`、`cua_energon_adapter.py` 和 `import_qwen35_vlm_checkpoint.py` 原先放在 `shared/`，现在各复制到一版目录里。这样复制 `atomic/` 或 `complex/` 到其他项目时不会找不到仓库根目录的文件。代价是这三个文件有少量重复；若以后修改其中一版，需要检查另一版是否也需要同步。

`import_qwen35_vlm_checkpoint.py` 是**可选准备工具**，训练脚本不会自动运行。它将本地 HF 格式的模型权重导入 Megatron 格式。训练脚本使用环境变量 `INIT_CKPT` 指向准备好的 Megatron checkpoint。

## 使用者需要准备什么

1. Linux 与 CUDA 训练环境，八张 A100 80GB 或能够承载同样并行设置的 GPU。
2. 外部 `Megatron-Bridge`、`Megatron-LM`、Energon、PyTorch 等依赖。记录的 Bridge commit 为 `5cb3444c43f7499cf3872b2d46870cf8bc2e00ce`。
3. 本地 Qwen3.5-4B HF 模型目录，提供配置、processor、tokenizer 和模型权重。
4. 本路线 Megatron 格式 `INIT_CKPT`：atomic 为 base；complex 为匹配路线的 atomic `iter_0000422`。
5. 足够磁盘空间和访问公开 HF 数据集的网络。脚本先下载图片到本地，再开始训练，不逐张远程流式读取。

复制一个目录后，在该目录内运行 `bash train.sh func` 或 `bash train.sh gui`。具体环境变量和示例见各自 README。下载器会把 `main` 解析为当时的 commit；正式复现实验建议把 `DATA_REVISION` 固定为具体 commit。

## 当前验证边界

- HF 上四份训练 JSONL 的 SHA256 与源端一致；atomic 的 530,472 条唯一图片引用路径均存在；complex 的 208,791 个源目录文件相对路径与源端一致。
- 每版下载器的离线测试、Python 编译和 Bash 语法检查已通过。
- 尚未在全新机器上下载整套数据并完成端到端八卡训练。因此“目录可以独立使用”是代码引用和输入路径层面的保证，不是新环境训练已成功的声明。
