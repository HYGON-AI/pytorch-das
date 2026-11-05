### C86-HPCKit 使用说明

> 用 **c86-hpckit_20261010.beta**（下称 HPCKit）配合 `scripts/build_hcu_hml.sh`，在本仓库编译出
> **HML + HygonDNN**（纯海光数学库，不依赖 MKL）版 PyTorch wheel。

#### 1. 获取与安装

| 资源 | 地址 |
|---|---|
| HPCKit 下载 | https://download.sourcefind.cn:65024/8/main/c86-devkit/latest |

包约 **876 MB**，下载后按页面 checksum 校验（`sha256sum` / `md5sum`）。

```bash
tar -xzf c86-hpckit_20261010.beta.tar.gz && cd c86-hpckit_20261010

sudo bash install.sh                              # 默认 /opt/hygon（需 root）
bash install.sh --install-dir /home/<user>/tools  # 指定安装 /home/<user>/tools/hygon (普通用户)
```

#### 2. 环境配置

```bash
source /opt/hygon/hpc-setenv.sh hgcc     # 参数必须是 hcc 或 hgcc（不传会交互式提问）
source /opt/hygon/hpc-unsetenv.sh        # 还原
```

#### 3. 编译

```bash
bash scripts/build_hcu_hml.sh --hml-home /opt/hygon/hml/hgcc --hdnn-home /opt/hygon/hdnn
```

参数说明:

| 环境变量 | 作用 | 默认 |
|---|---|---|
| `SKIP_PIP_BUILD=1` | 跳过脚本内 pip 依赖安装（离线/已备好） | 不跳过 |
| `INSTALL_WHEEL=0` | 只出 wheel 不安装（默认会**覆盖**当前环境 torch） | `1` |
| `MANYLINUX=1` | 出包后跑 `build_hcu_hml_manylinux.sh`（需 `patchelf`/`auditwheel`，且需 wheel 已安装） | `0` |
| `EXTRA_EXCLUDES` | auditwheel `--exclude` 列表（`MANYLINUX=1` 时默认排除 HML/HDNN 运行时库） | 见脚本 |
| `USE_MKLDNN` | 是否编译 MKLDNN/HDNN 路径 | `1` |
| `MAX_JOBS` | 编译并行度（OOM 时降低） | 不设，用满核 |

#### 4. 安装、校验与回滚

```bash
# 安装（不拉依赖，避免动到当前环境的依赖）
pip3 install --no-deps --force-reinstall dist/torch-*.whl
```

