# Frequency-Time Diffusion with Neural Cellular Automata

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Status](https://img.shields.io/badge/Status-Complete-success)
![License](https://img.shields.io/badge/License-MIT-green)

> **State-of-the-Art Implementation:** A hybrid generative model combining Spatial Cellular Automata with Frequency-Domain Diffusion. This optimized implementation achieves an **FID of 34.57**, outperforming the original research paper baseline (FID 43.86) while using fewer parameters.

---

##  Project Overview

Standard Neural Cellular Automata (NCA) excel at generating local textures (like biological tissue) but struggle with global structures (like faces or organ shapes). This project implements **FourierDiffNCA**, a dual-stream architecture that solves this by coupling:

1.  **Spatial Branch (Refiner):** A standard NCA that evolves local pixel interactions to generate high-fidelity textures.
2.  **Fourier Branch (Planner):** A frequency-domain diffusion model that plans global structure.

### Key Innovation: The "Bottleneck" Optimization
Unlike the original paper which used a symmetric architecture (~1.1M params), this project introduces a **64-channel Latent Bottleneck** in the Fourier branch. This forces the model to compress global information, acting as a regularizer that filters out high-frequency noise.

**Result:**
* **Parameter Count:** Reduced by ~11% (**0.98M** vs 1.1M).
* **Performance:** Improved FID score by **~21%** (34.57 vs 43.86).
* **Efficiency:** Optimized for consumer hardware (RTX 5090) using Gradient Checkpointing and Mixed Precision (AMP).

---

##  Results & Benchmarks

We evaluated the model on the **CelebA** dataset ($64 \times 64$) and compared it against the baseline architecture reported in the literature.

| Model Variant | Parameters | FID Score ($\downarrow$) | KID Score ($\downarrow$) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Optimized (Ours)** | **~0.98M** | **34.57** | **0.0157** | **Best Performance** |
| Paper Baseline | ~1.1M | 43.86 | 0.0220 | Original Literature |
| Heavy Variant | ~1.7M | 50.26 | 0.0303 | Overfitting to noise |
| Naive Baseline | ~1.1M | 65.38 | 0.0463 | Unoptimized impl. |

### Visualizations
<img width="971" height="596" alt="image" src="https://github.com/user-attachments/assets/0ae57659-bd27-408a-b3dc-5e2080964bf5" />




---

##  Architecture

The system consists of two coupled Neural Cellular Automata:

1.  **Input:** Noisy Image ($x_t$) at step $t$.
2.  **Phase 1 (Global Planning):**
    * $FFT(x_t) \rightarrow$ Crop Low Freqs $\rightarrow$ **Bottleneck NCA (64 ch)** $\rightarrow$ IFFT.
    * Produces a "Semantic Guide" (Spatial Map of global structure).
3.  **Phase 2 (Local Refinement):**
    * The **Spatial NCA (96 ch)** is initialized with the Semantic Guide.
    * It runs for 20 steps to denoise the image and generate texture.

---

##  Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/frequency-time-diffusion-nca.git](https://github.com/yourusername/frequency-time-diffusion-nca.git)
    cd frequency-time-diffusion-nca
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

##  Usage

### 1. Training
To train the model from scratch on CelebA or Medical datasets:
```bash
python main.py#The script will automatically detect your GPU (cuda) and begin the training loop.

```
##  References

This project is an optimized implementation based on the concepts from:

* **Frequency-Time Diffusion with Neural Cellular Automata (2024)**
    * *Authors:* John Kalkhof, Arlene Kühn, Yannik Frisch, Anirban Mukhopadhyay
    * *Paper:* [arXiv:2401.06291](https://arxiv.org/abs/2401.06291)
* *Denoising Diffusion Probabilistic Models (DDPM)*
* *Growing Neural Cellular Automata (Distill.pub)*
