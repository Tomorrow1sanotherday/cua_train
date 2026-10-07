# CUA 训练代码结构说明

## 核心结论

`atomic/` 和 `complex/` 是两个独立的训练目录。每个目录都带自己的 `setup.sh`、`run.sh`、`train.sh`、数据下载与校验、CUA 图文适配器、训练入口和测试；没有 `shared/`。完整数据放在 Hugging Face，GitHub 不存放图片或 JSONL。

**公开版不使用容器。**A800 的训练方式是宿主机 Python 虚拟环境加 `PYTHONPATH` 指向 Megatron-Bridge 源码。公开版按此思路：`setup.sh` 在当前版本的 `.runtime/` 创建隔离虚拟环境，取得固定源码和依赖；`run.sh` 配好环境后调用 `train.sh`，由八卡 `torchrun` 开始训练。用户需有正常的 NVIDIA 驱动、CUDA 开发工具链（`nvcc`）、八张适用 GPU、Git/编译器/Python 3/pip，并提供匹配的 Megatron checkpoint。

```text
atomic/ 或 complex/
  setup.sh                    首次取得 Bridge/Megatron-LM、安装依赖、编译 TE/Apex
  bridge-training.patch       服务器训练使用的三处 Bridge 补丁
  runtime-requirements.txt    关键 Python 依赖版本
  build_apex_fused.py         编译 PyTorch 对应的 Apex 权重梯度扩展
  run.sh                      宿主机统一入口：环境 + check/smoke/train
  train.sh                    下载/校验数据与模型，然后 torchrun
  runtime_check.py            GPU、数据、图片、模型和 checkpoint 预检
  download_data.py            从 HF 下载或校验本地数据
  download_model.py           可选下载公开 Qwen3.5-4B
  cua_energon_adapter.py      CUA JSONL 和图片转 Qwen 训练样本
  import_qwen35_vlm_checkpoint.py  可选 HF 权重导入工具
```

`atomic/` 额外有 `run_qwen35_cua_1ep.py`；`complex/` 额外有 `run_slides01_sft.py`、`action_window_provider.py` 和 `action_window_encoder.py`。后者把一条 Slides01 轨迹展开成逐 action 样本，并只让目标 action 与 EOS 参与 loss。Function/GUI 始终独立训练；complex 从同路线 atomic `iter_0000422` 开始。

## 新机器执行链

```text
bash atomic/run.sh func check
  -> setup.sh 在 atomic/.runtime/ 建 Python 3.12 虚拟环境
  -> 克隆固定 Bridge commit 与 Megatron-LM 子模块，应用训练补丁
  -> 安装 PyTorch 2.10/Transformers 5.8.1/Energon 7.4.0
  -> 编译固定源码的 Transformer Engine 和 Apex fused 扩展
  -> 校验 JSONL 哈希、索引、示例图片、模型 processor、checkpoint 和 8 GPU

bash atomic/run.sh func smoke
  -> 同样预检，只做 1 次 optimizer 更新，不保存 checkpoint

bash atomic/run.sh func train
  -> 原记录的 422 次更新与保存点
```

Complex 的 `run.sh func|gui check|smoke|train` 流程相同，但读最新 Slides01 v4 action-window 数据；三轮训练的保存点和监督方式由本版 `train.sh`/encoder 决定。若设置 `DATA_ROOT`，校验用户提供的本地数据；不设置则从公开 HF 下载。`MODEL_DIR` 可指定本地模型；不设置则下载公开 `Qwen/Qwen3.5-4B`。模型目录不代替 `INIT_CKPT`，训练初始参数仍来自用户提供的 Megatron checkpoint。

## 验证边界

- HF 四份训练 JSONL 的 SHA256 与源端一致；atomic 的 530,472 个唯一图片引用路径存在，complex 的 208,791 个源目录文件路径与源端一致。
- A800 宿主机已在独立 `/tmp` 虚拟环境用 CUDA 12.4 工具链完成固定 Bridge 补丁、PyTorch 2.10、源码编译 TE、Apex fused 扩展和 Python 3.12 数据 helper，并通过训练模块导入；重复运行 `setup.sh` 立即返回 `RUNTIME_READY`。
- 先前的容器检查验证过四条路线能读取真实数据、图片、processor、checkpoint 并看见八卡；这不能替代原生宿主机训练验证。
- 原生 `run.sh ... check` 和真实 optimizer 更新仍需在同时具备完整数据、匹配 checkpoint、空闲八卡的宿主机执行。A800 目前没有 atomic 完整数据且八卡被占用，DGX 宿主机没有完整 nvcc；因此尚不能声称全新服务器端到端训练已跑通。
