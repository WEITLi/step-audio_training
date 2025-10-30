# Step-Audio-EditX
<p align="center">
  <img src="assets/logo.png"  height=100>
</p>

<div align="center">
  <a href="https://huggingface.co/stepfun-ai/Step-Audio-TTS-3B"><img src="https://img.shields.io/static/v1?label=Step-Audio-TTS-3B&message=HuggingFace&color=yellow"></a> &ensp;
</div>

## 🔥🔥🔥 News!!
* Feb 17, 2025: 👋 We release the technical report of [Step-Audio](https://arxiv.org/abs/2502.11946).

## Table of Contents

1. [Introduction](#1-introduction)
2. [Model Summary](#2-model-summary)
3. [Model Download](#3-model-download)
4. [Model Usage](#4-model-usage)
5. [Benchmark](#5-benchmark)
6. [Online Engine](#6-online-engine)
7. [Examples](#7-examples)
8. [Acknowledgements](#8-acknowledgements)
9. [License Agreement](#9-license-agreement)
10. [Citation](#10-citation)

## 1. Introduction

## 2. Model Summary

## 3. Model Download
### 3.1 Huggingface
| Models   | Links   |
|-------|-------|
| Step-Audio-EditX | [🤗huggingface](https://huggingface.co/stepfun-ai/Step-Audio-EditX) |

### 3.2 Modelscope
| Models   | Links   |
|-------|-------|
| Step-Audio-EditX | [modelscope](https://modelscope.cn/models/stepfun-ai/Step-Audio-EditX) |

## 4. Model Usage
### 📜 4.1  Requirements
The following table shows the requirements for running Step-Audio model (batch size = 1):

|     Model    |  Setting<br/>(sample frequency) | GPU Minimum Memory  |
|------------|--------------------------------|----------------|
| Step-Audio-EditX   |        41.6Hz          |       8GB        |

* An NVIDIA GPU with CUDA support is required.
  * The model is tested on a four A800 80G GPU.
  * **Recommended**: We recommend using 4xA800/H800 GPU with 80GB memory for better generation quality.
* Tested operating system: Linux

### 🔧 4.2 Dependencies and Installation
- Python >= 3.10.0 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
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

After downloading the models, where_you_download_dir should have the following structure:
```
where_you_download_dir
├── Step-Audio-Tokenizer
├── Step-Audio-EditX
```

#### Run with Docker

You can set up the environment required for running Step-Audio using the provided Dockerfile.

```bash
# build docker
docker build . -t step-audio-editx

# run docker
docker run --rm --gpus all \
    -v /your/code/path:/app \
    -v /your/model/path:/model \
    -p 7860:7860 \
    step-audio-editx
```


#### Launch Web Demo
Start a local server for online inference.
Assume you have 4 GPUs available and have already downloaded all the models.

```bash
# Step-Audio-EditX demo
python app.py --model-path where_you_download_dir --model-source local 
```

## 5. Benchmark

## 6. Online Engine
The online version of Step-Audio can be accessed from app version of [跃问](https://yuewen.cn), where some impressive examples can be found as well.

<img src="./assets/yuewen.jpeg" width="200" alt="QR code">

## 7. Examples

## 8. Acknowledgements

Part of the code for this project comes from:
* [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
* [transformers](https://github.com/huggingface/transformers)
* [FunASR](https://github.com/modelscope/FunASR)

Thank you to all the open-source projects for their contributions to this project!
## 9. License Agreement

+ The use of weights for Step Audio related models requires following license in [Step-Audio-Tokenizer](https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer/tree/main) and [Step-Audio-EditX](https://huggingface.co/stepfun-ai/Step-Audio-EditX/tree/main)

+ The code in this open-source repository is licensed under the [Apache 2.0](LICENSE) License.

## 10. Citation

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=stepfun-ai/Step-Audio&type=Date)](https://star-history.com/#stepfun-ai/Step-Audio&Date)
