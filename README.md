DiffNCA for Medical Imaging

A PyTorch implementation of Diffusion Neural Cellular Automata (DiffNCA) for high-fidelity synthesis of Breast Cancer pathology images.

This repository implements DiffNCA and FourierDiffNCA, generative models that combine the iterative local updates of Cellular Automata with the stable training objective of Denoising Diffusion Probabilistic Models (DDPM). The project is optimized for high-resolution medical image generation using Gradient Checkpointing and Mixed Precision (AMP) to fit large hidden states on consumer GPUs ( RTX 5090)

 Key Features
 
 Dual Architectures:
 
   DiffNCA: Standard image-space Neural Cellular Automata for texture synthesis.

  FourierDiffNCA: A hybrid model operating in both frequency (Fourier) and image domains for better global coherence.

  Medical & General Domains:

  BCSS (Breast Cancer Semantic Segmentation): Custom pipeline for processing large H&E histology slides into training patches.

   CelebA: Built-in support for face generation benchmarks.
   

  Optimized Training:

   Gradient Checkpointing: Reduces VRAM usage by ~90% (enabling $h=512$ hidden states).

   Torch Compile: Uses PyTorch 2.0 kernel fusion for faster NCA loops.

   Mixed Precision (AMP): Faster training on NVIDIA Tensor Cores.

 

  

    
