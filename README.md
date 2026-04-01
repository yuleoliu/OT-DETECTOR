# OT-DETECTOR: Delving into Optimal Transport for Zero-shot Out-of-Distribution Detection

This codebase reproduces the key results from our paper, "[OT-DETECTOR: Delving into Optimal Transport for Zero-shot Out-of-Distribution Detection]", presented at IJCAI 2025.

### Abstract

Out-of-distribution (OOD) detection is crucial for ensuring the reliability and safety of machine learning models in real-world applications. While zeroshot OOD detection, which requires no training on in-distribution (ID) data, has become feasible with the emergence of vision-language models like CLIP, existing methods primarily focus on semantic matching and fail to fully capture distributional discrepancies. To address these limitations, we propose OT-DETECTOR, a novel framework that employs Optimal Transport (OT) to quantify both semantic and distributional discrepancies between test samples and ID labels. Specifically, we introduce cross-modal transport mass and transport cost as semantic-wise and distribution-wise OOD scores, respectively, enabling more robust detection of OOD samples. Additionally, we present a semantic-aware content refinement (SaCR) module, which utilizes semantic cues from ID labels to amplify the distributional discrepancy between ID and hard OOD samples. Extensive experiments on several benchmarks demonstrate that OTDETECTOR achieves state-of-the-art performance across various OOD detection tasks, particularly in challenging hard-OOD scenarios.

### Illustration

![Framework](figs/framework.pdf)

### Installation
conda create -n ot-detector python=3.8
conda activate ot-detector
pip install -r requirements.txt

### Data Preparation

please refer to [Huang et al. 2021](https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset) for the preparation of the following datasets:Imagenet, iNaturalist, SUN, Places, Texture.

The overall file structure is as follows:
```
OT-DETECTOR
|-- dataset
    |-- ImageNet 
    |-- iNaturalist
    |-- dtd
    |-- SUN
    |-- Places
    ...
```

### Quick Start
```python
    torchrun --nproc_per_node=8 dist_eval.py
```
We also provide the already extracted feature similarity files [Google Drive](https://drive.google.com/drive/folders/1jFMm9mCbpPL3xhyQLbV7tPmzkp9MWwfX?usp=drive_link). Put them in the crop_features folder to run:
```python    
    torchrun --nproc_per_node=8 dist_eval.py --load True
```

### Acknowledgments

This project's implementation of the Maximum Concept Matching (MCM) method is built upon and inspired by the official code repository:

**MCM Official Repository:** [deeplearning-wisc/MCM](https://github.com/deeplearning-wisc/MCM)

We sincerely thank the authors for making their excellent work and implementation publicly available.