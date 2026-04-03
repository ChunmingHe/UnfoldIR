<p align=center><img src="figs/logo.png" width="200px"> </p>

# <p align=center> `UnfoldIR` </p>

<b><p align=center>
<a href='YOUR_PAPER_LINK'><img src='https://img.shields.io/badge/ArXiv-XXXX.XXXXX-red'></a>
<!-- 如果有会议/期刊信息，可以保留 -->
CVPR / ICCV / ECCV / TIP / TCSVT / etc.
</p></b>

**UnfoldIR: [Your full paper title here] [[Paper]](YOUR_PAPER_LINK)**

[Author1](AUTHOR1_HOMEPAGE), [Author2](AUTHOR2_HOMEPAGE), [Author3](AUTHOR3_HOMEPAGE), Author4, Author5

#### 🔥🔥🔥 News
- **2026-XX-XX:** We release the code and pretrained models.
- **2026-XX-XX:** We update the training and testing scripts.
- **2026-XX-XX:** Paper is available on arXiv.
- **2026-XX-XX:** This repository is created.

> **Abstract:**  
> Image restoration under complex degradations remains challenging due to the entanglement of degradation factors and image content. To address this issue, we propose **UnfoldIR**, an unfolding-based image restoration framework that integrates model-inspired optimization priors with learnable deep architectures. Specifically, our method progressively refines latent representations through multi-stage restoration modules, enabling more stable degradation disentanglement and more faithful detail recovery. Extensive experiments on multiple benchmark datasets demonstrate that UnfoldIR achieves competitive or state-of-the-art performance in both quantitative metrics and visual quality.

![](figs/framework.png)

---

## 🔗 Contents

- [x] [Requirements](#-requirements)
- [x] [Training](#-training)
- [x] [Testing](#-testing)
- [x] [Results](#-results)
- [x] [Citation](#-citation)
- [x] [Acknowledgements](#-acknowledgements)

---

## 📦 Requirements

Pretrained Models: [Google Drive](YOUR_GOOGLE_DRIVE_LINK)

### ⚙️ Dependencies

- Python 3.9
- PyTorch 2.0.1
- CUDA 11.8
- NVIDIA GPU

### Initialize Environment

```bash
git clone https://github.com/YOUR_USERNAME/UnfoldIR.git
cd UnfoldIR

conda create -n UnfoldIR python=3.9
conda activate UnfoldIR

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
python setup.py develop
