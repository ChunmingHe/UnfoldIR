<p align=center><img src="figs/logo.png" width="200px"> </p>

# <p align=center> `UnfoldIR` </p>

<b><p align=center>
<b><p align=center> <a href='https://arxiv.org/abs/2505.06683'><img src='https://img.shields.io/badge/ArXiv-2505.06683-red'></a>
CVPR26 ✨</p></b>

**UnfoldIR: Rethinking Deep Unfolding Network in Illumination Degradation Image Restoration [[Paper]](https://arxiv.org/abs/2505.06683)**

[Chunming He](https://chunminghe.github.io/), Rihan Zhang, Fengyang Xiao, Chengyu Fang, Longxiang Tang, Rui Zhang, and Sina Farsiu

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

- [x] [Requirements](https://github.com/ChunmingHe/Reti-Diff/blob/main/README.md#-requirements)
- [x] [Training](https://github.com/ChunmingHe/Reti-Diff/blob/main/README.md#-training)
- [x] [Testing](https://github.com/ChunmingHe/Reti-Diff/blob/main/README.md#-testing)
- [x] [Results](https://github.com/ChunmingHe/Reti-Diff/blob/main/README.md#-results)
- [x] [Citation](https://github.com/ChunmingHe/Reti-Diff/blob/main/README.md#-citation)
- [x] [Acknowledgements](https://github.com/ChunmingHe/Reti-Diff/blob/main/README.md#-acknowledgements)

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


## 🧱 Training

1. Put the Train Datasets into Datasets folder.

2. Modify the config files in options folder and specific shell scripts.

3. Run the shell scripts

```bash
# LLIE
sh trainS1_LLIE.sh
sh trainS2_LLIE.sh

# UIE
sh trainS1_UIE.sh
sh trainS2_UIE.sh

# Backlit
sh trainS1_Backlit.sh
sh trainS2_Backlit.sh
```

## ⚡️ Testing

1. Put the Test Datasets into Datasets folder.

2. Put the Pretrained Models into pretrained_models folder.

3. Run the following command to test the model:

```bash
# LLIE
sh test_LLIE_syn.sh
or 
sh test_LLIE_real.sh

# UIE
sh test_UIE_LSUI.sh
or
sh test_UIE_UIEB.sh

# Backlit
sh test_Backlit.sh
```

## 🔍 Results

We achieved state-of-the-art performance on *low light image enhancement*, *underwater image enhancement*, *backlit image enhancement* and corresponding downstream tasks. More results can be found in the paper.

<details>
<summary>Quantitative Comparison (click to expand)</summary>

- Results in Table 1 of the main paper
  <p align="center">
  <img width="900" src="figs/table-1.png">
	</p>
- Results in Table 2-3 of the main paper
  <p align="center">
  <img width="900" src="figs/table-2-3.png">
	</p>
- Results in Table 6-9 of the main paper
  <p align="center">
  <img width="900" src="figs/table-6-7-8-9.png">
	</p>
  </details>

<details>
<summary>Visual Comparison (click to expand)</summary>

- Results in Figure 3 of the main paper
  <p align="center">
  <img width="900" src="figs/llie.jpeg">
	</p>
- Results in Figure 4 of the main paper
  <p align="center">
  <img width="900" src="figs/uie.jpeg">
	</p>
- Results in Figure 5 of the main paper
  <p align="center">
  <img width="900" src="figs/backlit.jpeg">
	</p>
  </details>


## 📎 Citation

If you find the code helpful in your research or work, please cite the following paper(s).

```
@article{he2025reti,
  title={Reti-Diff: Illumination Degradation Image Restoration with Retinex-based Latent Diffusion Model},
  author={He, Chunming and Fang, Chengyu and Zhang, Yulun and Li, Kai and Tang, Longxiang and You, Chenyu and Xiao, Fengyang and Guo, Zhenhua and Li, Xiu},
  journal={ICLR},
  year={2025}
}
```
