# PINNSR-DA
## Overview
Granular material has significant implications for industrial and geophysical processes. A long-lasting challenge, however, is seeking a unified rheology for its solid- and liquid-like behaviors under quasi-static, inertial, and even unsteady shear conditions. Here, we present a white-box machine learning framework to discover the hidden governing equation of sheared granular materials. The framework, PINNSR-DA, addresses noisy discrete particle data via physics-informed neural networks with sparse regression (PINNSR) and ensures dimensional consistency via machine learning-based dimensional analysis (DA). Applying PINNSR-DA to our discrete element method simulations of oscillatory shear flow, a general differential equation is found to govern the effective friction across steady and transient states. The equation consists of three interpretable terms, accounting respectively for linear response, nonlinear response and energy dissipation of the granular system, and the coefficients depends primarily on a dimensionless relaxation time, which is shorter for stiffer particles and thicker flow layers. This work pioneers a pathway for discovering physically interpretable governing laws in granular systems and can be readily extended to more complex scenarios involving jamming, segregation, and fluid-particle interactions.

This reposititory provides data and supplementary codes for the following work: 

- Xu Han, Lu Jing, Chung-Yee Kwok, Gengchao Yang and Yuri Dumaresq Sobral. **"Data-driven discovery of physically-interpretable governing equations for sheared granular materials."** arXiv preprint arXiv:2509.14518 (2025).

## System Requirements
### Hardware requirements
This work was conducted using a workstation with the following configurations:
- Intel 14th Gen Core i7-14700KF processor (20 cores/28 threads, up to 5.6GHz);
- NVIDIA GeForce RTX 4090 graphics card.

### Software requirements
- **OS Requirements**:
The package has been tested on Windows 11.
- **Software dependencies**:
This work was implemented on Pytorch 1.12.0 in Python, along with its associated libraries torchaudio 0.12.0 and torchvision 0.13.0, with GPU acceleration. Other detailed pip requirements can be found in [requirements.txt](./requirements.txt).

## Supplemtary Information
- **Supplementary Information for the Main Text**
	- This Supplementary Information serves as a detailed companion to the main text and is referred to therein as "SI". It comprises four sections: Supplementary Algorithm, Supplementary Results, Supplementary Discussion, and Supplementary References.
- **Supplementary Video**
	- This supplementary video is provided to facilitate readers' understanding of oscillatory simple shear. It demonstrates two loading scenarios: sinusoidal loading and Heaviside loading.
    - A clearer and higher-quality version of the video is available at the following [YouTube link](https://www.youtube.com/watch?v=Zzx57moZebE).
