# Step-Audio-EditX
<p align="center">
  <img src="assets/logo.png"  height=100>
</p>
<div align="center">
  <a href="https://huggingface.co/stepfun-ai/Step-Audio-TTS-3B"><img src="https://img.shields.io/static/v1?label=Step-Audio-TTS-3B&message=HuggingFace&color=yellow"></a> &ensp;
</div>

## 🔥🔥🔥 News!!
* 2025年2月17日: 👋 发布了技术报告[Step-Audio-Report](https://arxiv.org/abs/2502.11946)。

## Table of Contents

1. [介绍](#1-介绍)
2. [模型组成](#2-模型组成)
3. [模型下载](#3-模型下载)
4. [模型使用](#4-模型使用)
5. [基准](#5-基准)
6. [在线引擎](#6-在线引擎)
7. [样例](#7-样例)
8. [致谢](#8-致谢)
9. [协议](#9-协议)
10. [引用](#10-引用)

## 1. 介绍

## 2. 模型组成

## 3. 模型下载
### 3.1 Huggingface
| 模型   | 链接   |
|-------|-------|
| Step-Audio-EditX | [🤗huggingface](https://huggingface.co/stepfun-ai/Step-Audio-EditX) |

### 3.2 Modelscope
| 模型   | 链接   |
|-------|-------|
| Step-Audio-EditX | [modelscope](https://modelscope.cn/models/stepfun-ai/Step-Audio-EditX) |

## 4. 模型使用
### 📜 4.1  要求
下表列出了运行Step-Audio模型（batch size=1）所需的配置要求:

|     模型    |  Setting<br/>(采样率) | GPU最低显存  |
|------------|--------------------------------|----------------|
| Step-Audio-EditX   |        41.6Hz          |       8GB        |

* 需要支持CUDA的NVIDIA显卡.
  * 模型在4块显存为80GB的A800系列NVIDIA显卡上进行测试.
  * **推荐**: 为确保最佳生成质量，建议使用4块显存为80GB的A800/H800系列NVIDIA显卡.
* 测试采用的操作系统: Linux

### 🔧 4.2 依赖项与安装
- Python >= 3.10.0 (推荐使用 [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.3-cu121](https://pytorch.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

```bash
git clone https://github.com/stepfun-ai/Step-Audio-EditX.git
conda create -n stepaudioedit python=3.10
conda activate stepaudioedit

cd Step-Audio
pip install -r requirements.txt

git lfs install
git clone https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer
git clone https://huggingface.co/stepfun-ai/Step-Audio-EditX

```

下载模型后，where_you_download_dir应包含以下结构：
```
where_you_download_dir
├── Step-Audio-Tokenizer
├── Step-Audio-EditX
```

#### Docker 运行环境

使用 `docker` 创建 `Step-Audio` 运行时所需要的环境

```bash
# 构建 docker 镜像
docker build . -t step-audio-editx

# 运行 docker
docker run --rm --gpus all \
    -v /your/code/path:/app \
    -v /your/model/path:/model \
    -p 7860:7860 \
    step-audio-editx
```

#### 启动网页演示
启动本地服务器以进行在线推理。
假设您已配备4块GPU且已完成所有模型的下载。

```bash
# Step-Audio-EditX demo
python app.py --model-path where_you_download_dir --model-source local 
```

## 5. 基准


## 6. 在线引擎
Step-Audio 的在线版本可以通过[跃问](https://yuewen.cn) 的应用程序访问，其中还可以找到一些惊喜的示例。

<img src="./assets/yuewen.jpeg" width="200" alt="QR code">

## 7. 样例

## 8. 致谢

本项目的部分代码来自：
* [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
* [transformers](https://github.com/huggingface/transformers)
* [FunASR](https://github.com/modelscope/FunASR)

感谢以上所有开源项目对本项目开源做出的贡献！
## 9. 协议

+ Step-Audio 相关模型的权重使用协议请分别需要按照[Step-Audio-Tokenizer](https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer/tree/main) 和 [Step-Audio-EditX](https://huggingface.co/stepfun-ai/Step-Audio-EditX/tree/main) 里面的协议进行遵守

+ 本开源仓库的代码则遵循 [Apache 2.0](LICENSE) 协议。

## 10. 引用

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=stepfun-ai/Step-Audio-EditX&type=Date)](https://star-history.com/#stepfun-ai/Step-Audio-EditX&Date)
