# PyTorch for Hygon

## 开源声明

本仓库基于官方 PyTorch 项目，包含海光（Hygon）维护的定制修改。

- 上游项目：PyTorch
- 上游源码：https://github.com/pytorch/pytorch
- 上游文档：https://pytorch.org/docs/stable/
- 上游 README 保留在本仓库：[README_ORIGIN.md](README_ORIGIN.md)
- 许可证：BSD-3-Clause

`README_ORIGIN.md` 作为上游参考副本保留，用于对照与审查。其内容保持上游原文不变，不代表本仓库对外发布的说明。

官方 PyTorch 的版权、许可条款与免责声明保留在 [LICENSE](LICENSE) 中，第三方声明见 [NOTICE](NOTICE)。

Modified by Hygon Information Technology Co., Ltd.
Copyright (c) 2026 Hygon Information Technology Co., Ltd.

## 修改范围

本仓库中 Hygon 的修改旨在使 PyTorch 适配海光软件与硬件环境，包括但不限于 HCU 后端集成、构建与打包调整、算子与内核适配，以及相关文档和元数据更新。

除源码文件或第三方声明中另有说明外，源自上游 PyTorch 的文件仍受上游 PyTorch 许可条款约束。

## PyTorch 简介

PyTorch 是一个基于 Python 的开源机器学习框架，以其强大的 GPU 加速张量计算能力和动态计算图技术，在深度学习领域占据重要地位。它广泛应用于计算机视觉、自然语言处理等人工智能关键领域。PyTorch 之所以备受青睐，得益于其易用性、灵活性和高效性，尤其适合快速原型开发、学术研究以及生产环境中的深度学习任务。

本软件栈中的 PyTorch 版本，不仅保证了 PyTorch 核心功能在 HCU 加速卡上的可用性，还针对 HCU 特有的硬件架构进行了深度定制优化。这使得开发者能够以极低的成本，轻松实现应用程序在 HCU 加速卡上的快速迁移和性能提升。

## 版本约束

1. 暂不支持的官方版本或功能

   **更高版本**：当前仓库开源 PyTorch 2.12.0 版本，更高版本将在后续陆续开源，敬请期待。

2. PyTorch 软件版本配套关系

   | PyTorch 版本 | HCU 适配版 PyTorch 软件包版本 | HCU 加速卡型号       | DTK 版本 |
   | ------------ | ----------------------------- | -------------------- | -------- |
   | 2.12.0       | 2.12.0+das.opt1.dtk2604       | K100_AI、bw1000 等  | 26.04    |

## 前置条件

使用本 PyTorch 需要参考[《HCU 新手入门教程》](https://developer.hpccube.com/gitbook/dcu_tutorial/index.html)在主机系统安装以下组件：

- HCU 驱动程序
- DTK
- Docker 引擎

## 使用命令安装 PyTorch

工具安装使用 pip 方式，从[光源社区 - DAS](https://cancon.hpccube.com:65024/4/main/pytorch)中下载此工具的安装包。注意与 Python 版本匹配。

```
pip3 install torch-xxx.whl
```

## 验证

安装完成之后，可通过以下指令验证是否安装成功，指令执行后会显示当前 PyTorch 的版本号。

```
python -c "import torch; print(torch.__version__)"
```

## 从源码编译（HCU/DTK）

### 环境要求

- 编译镜像：Rocky Linux 8（CentOS 8）及以上，自带 GCC 工具链，无需额外激活
- Python 3.10（脚本默认 `PYTHON_VERSION=3.10.12`）
- DTK 26.04（脚本自动从 `DTK_URL` 下载并部署到 `/opt/dtk`）
- 编译机需可访问代码仓库（Git 服务）与 DTK/三方库下载服务器

### 编译步骤

1. 拉取源码并进入仓库根目录（仓库地址与分支请以实际发布为准）：

   ```bash
   git clone -b <分支名> <仓库地址>
   cd pytorch
   ```

2. 执行编译脚本（脚本会自动完成：拉取/切换分支、下载部署 DTK、初始化三方库子模块并应用补丁、安装 Python 依赖、hipify 转码、`setup.py` 编译、wheel 与 manylinux 出包）：

   ```bash
   bash scripts/build_hcu.sh
   ```

   如需跳过 MAGMA（`USE_MAGMA=0`，不下载、不链接 MAGMA）：

   ```bash
   bash scripts/build_hcu.sh --no-magma
   ```

3. 编译产物：

   - wheel 包：`dist/torch-*.whl`
   - manylinux 包：`manylinux_*_x86_64` 目录下的 `torch-*.whl`

4. 安装并验证：

   ```bash
   pip3 install dist/torch-*.whl
   python -c "import torch; print(torch.__version__)"
   ```

### 编译脚本说明

编译脚本 `scripts/build_hcu.sh` 中的关键参数（代码仓库地址、分支、DTK/MAGMA 下载地址等）均在脚本开头变量区配置，便于每次编译前确认修改：

| 参数 | 说明 | 默认值 |
| ---- | ---- | ------ |
| `PYTORCH_REPO_URL` | PyTorch 代码仓库地址 | 由使用者指定 |
| `PYTORCH_BRANCH` | 编译分支 | 由使用者指定 |
| `DTK_URL` | DTK 安装包下载地址 | 由使用者指定 |
| `MAGMA_URL` | MAGMA 库下载地址（`--no-magma` 时忽略） | 由使用者指定 |
| `AMDGPU_TARGETS` | 编译目标算力架构 | `gfx906;gfx926;gfx928;gfx936;gfx938` |

### 三方库管理

三方库与上游 PyTorch 保持一致，以 Git 子模块方式管理。编译脚本会自动执行：

```bash
git submodule update --init --recursive
bash tools/third_party/apply_patches.sh
```

各三方库的许可证见其对应仓库。

## 建议阅读

有关 PyTorch 的更多信息，请参见：

- [AI 生态包](https://cancon.hpccube.com:65024/4/main/)
- [Model Zoo](https://sourcefind.cn/#/model-zoo/list)
- [HCU Toolkit](https://cancon.hpccube.com:65024/1/main)
- [驱动](https://cancon.hpccube.com:65024/6/main)
- [学习中心](https://developer.hpccube.com/study/)
- [论坛](https://forum.hpccube.com/)
- [PyTorch 2.12 官方文档](https://pytorch.org/docs/2.12/)

## 参考链接

- [官方 PyTorch 仓库](https://github.com/pytorch/pytorch)
- [官方 PyTorch 文档](https://pytorch.org/docs/stable/)
- [上游 README 原文](README_ORIGIN.md)
