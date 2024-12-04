--------------------------------------------------------------------------------

## PyTorch简介

PyTorch是一个基于Python的开源机器学习框架，以其强大的GPU加速张量计算能力和动态计算图技术，在深度学习领域占据重要地位。它广泛应用于计算机视觉、自然语言处理等人工智能关键领域。PyTorch之所以备受青睐，得益于其易用性、灵活性和高效性，尤其适合快速原型开发、学术研究以及生产环境中的深度学习任务。DAS软件栈中的PyTorch版本，不仅保证了PyTorch 核心功能在HCU加速卡的可用性,还针对HCU特有的硬件架构进行了深度定制优化。这使得开发者能够以极低的成本，轻松实现应用程序在HCU加速卡上的快速迁移和性能提升。

## 版本约束

1. 暂不支持的官方版本或功能

   **更高版本**：暂不支持PyTorch2.4以上版本

2. PyTorch软件版本配套关系

   | PyTorch版本 | HCU适配版PyTorch软件包版本                 | HCU加速卡型号              | DTK版本 |
   | ----------- | ------------------------------------------ | -------------------------- | ------- |
   | 2.3.0       | 2.3.0+das.opt2.dtk24043                    | Z100、Z100L、K100、K100_AI | 24.04.3 |
   | 2.1.0       | 2.1.0+das.opt2.dtk24043                    | Z100、Z100L、K100、K100_AI | 24.04.3 |
   | 2.3.0       | 2.3.0+das.opt1.dtk24042                    | Z100、Z100L、K100、K100_AI | 24.04.2 |
   | 2.1.0       | 2.1.0+das.opt1.dtk24042                    | Z100、Z100L、K100、K100_AI | 24.04.2 |
   | 2.1.0       | 2.1.0+das1.1.git3ac1bdd.abi1.dtk2404       | Z100、Z100L、K100、K100_AI | 24.04.1 |
   | 2.1.0       | torch-2.1.0+das1.0+git00661e0.abi0.dtk2404 | Z100、Z100L、K100、K100_AI | 24.04   |

## 前置条件

使用 DAS PyTorch需要参考[《HCU新手入门教程》](https://developer.hpccube.com/gitbook/dcu_tutorial/index.html)在主机系统安装以下组件:

- HCU驱动程序
- DTK
- Docker引擎

## 使用命令安装PyTorch

工具安装使用 pip 方式，从[光源社区 -DAS](https://cancon.hpccube.com:65024/4/main/pytorch) 中下载此工具的安装包。注意与 python 版本匹配

pip3 install torch-xxx.whl 

## 验证

安装完成之后，可通过以下指令验证是否安装成功,指令执行后会显示当前PyTorch的版本号。

```
python -c "import torch; print(torch.__version__)"
```

## 建议阅读

有关PyTorch的更多信息，请参见:

- [AI生态包](https://cancon.hpccube.com:65024/4/main/)
- [Model Zoo](https://sourcefind.cn/#/model-zoo/list)
- [HCU Toolkit](https://cancon.hpccube.com:65024/1/main)
- [驱动](https://cancon.hpccube.com:65024/6/main)
- [学习中心](https://developer.hpccube.com/study/)
- [论坛](https://forum.hpccube.com/)
- [PyTorch 2.3 官方文档](https://pytorch.org/docs/2.3/)