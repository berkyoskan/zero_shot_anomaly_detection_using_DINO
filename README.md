
# Zero-, One-, and Few-Shot Anomaly Detection Using Foundation Models

This repository contains the code and resources for **“Zero-, One-, and Few-Shot Anomaly Detection Using Foundation Models”**, a project for the 02456 Deep Learning course at DTU Compute (Fall 2025). In this work we evaluate how vision foundation models (primarily DINOv3) can detect industrial anomalies with minimal training data, and how optional foreground segmentation (PCA or SAM3) influence the performance.

### Authors
- Adams Ali Gills (s243894)  
- Berk Yozkan (s253820)  
- Mikolaj Zbigniew Dzwigalo (s253816)  
- Vladimir Salnikov (s252682)

### Overview
We evaluate anomaly detection in three data regimes:
- **Zero-shot**: text–image similarity with DINOv3 encoders.  
- **One-shot / Few-shot (PatchKNN)**: build a memory bank from several reference images, compute patch-level k-NN distances for test images, and upsample to heatmaps.  
- **Segmentation ablations**: compare no segmentation vs. PCA-based masks vs. SAM3 prompts to see when foreground masking helps.

Experiments are run on MVTec AD and a private dataset (concrete/wood walls) to probe robustness and data contamination concerns. Metrics include image-level AUROC/F1 and pixel-level AUROC/F1.

### Quick Preview
- **Space demo**: You can acess model demo on [hugging face](https://huggingface.co/spaces/V4ldeLund/AnomalyDetection)

