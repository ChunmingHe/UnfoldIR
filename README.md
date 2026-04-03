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
>Deep unfolding networks (DUNs) are widely employed in illumination degradation image restoration (IDIR), merging the interpretability of model-based approaches with the power of deep learning. 
However, their performance still lags behind state-of-the-art IDIR methods, primarily due to insufficient exploration of unfolding structures. The main challenges are: (1) lack of task-specific modeling, (2) weak integration of modern network designs, and (3) absence of DUN-tailored learning objectives.
To address this, we propose a novel DUN-based method, UnfoldIR, for IDIR tasks. 
It formulates a new IDIR model with regularization terms for illumination smoothing and texture enhancement, whose iterative optimization is unfolded into a multi-stage network containing two modules:
reflectance-assisted illumination correction (RAIC) and illumination-guided reflectance enhancement (IGRE). 
RAIC employs a visual state space (VSS) to extract non-local features, enforcing illumination smoothness, while IGRE introduces a frequency-aware VSS to globally align similar textures, enabling mildly degraded regions to guide the enhancement of details in more severely degraded areas. 
Furthermore, given the multistage structure, an inter-stage information consistency loss is proposed to enhance training stability and structural preservation, even under unsupervised settings.
Experiments verify our effectiveness across 5 IDIR tasks and 3 downstream tasks. Besides, our analysis of the intrinsic DUN mechanisms provides insights for future research.

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
