
from __future__ import annotations
import os
import cv2
import numpy as np
import torch
from segmenters.sam3 import SAM3Segmenter
from backbones import get_backbone
from dataset.dataloader import load_mvtec
from models.model_bank_knn import PatchKNNDetector
from evaluation.anomaly_evaluator import AnomalyEvaluator 

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup
    category = "bottle"
    root = "Desktop/datasets/MVTecAD/archive" 
    train_paths, test_paths = load_mvtec(category=category, root=root)
    test_paths = test_paths[::20]
    train_paths = train_paths[::20]
    print(f"{category}: {len(train_paths)} train, {len(test_paths)} test images")

    # 2. Initialize Evaluator with PRO support
    # We set compute_pro=True so we can calculate the region overlap
    evaluator = AnomalyEvaluator(pixel_subsample_rate=0.01, compute_pro=False)

    # 3. Model Init
    segmenter = SAM3Segmenter(text_prompt=category, device=device)
    backbone = get_backbone("dinov2_small")
    model = PatchKNNDetector(
        backbone=backbone,
        segmenter=segmenter,
        device=device,
        k_neighbors=1,
    )

    print("Fitting model...")
    model.fit(train_paths, n_ref=1)

    # 4. Evaluation Loop
    print(f"Starting evaluation on {len(test_paths)} images...")

    for i, path in enumerate(test_paths):
        # Predict
        image, amap, score = model.predict(path)

        # Ground Truth Logic
        is_anomaly = 0 if "good" in path else 1
        
        if is_anomaly == 0:
            gt_mask = np.zeros_like(amap)
        else:
            mask_path = path.replace("test", "ground_truth").replace(".png", "_mask.png")
            if os.path.exists(mask_path):
                gt_mask = cv2.imread(mask_path, 0)
                if gt_mask.shape != amap.shape:
                    gt_mask = cv2.resize(gt_mask, (amap.shape[1], amap.shape[0]), interpolation=cv2.INTER_NEAREST)
                gt_mask = (gt_mask > 0).astype(int)
            else:
                gt_mask = np.zeros_like(amap)

        # Update Evaluator
        evaluator.update(image_score=score, gt_label=is_anomaly, anomaly_map=amap, gt_mask=gt_mask)

        if i % 20 == 0:
            print(f"Processed {i}/{len(test_paths)}...")

    # 5. Compute & Print Results
    results = evaluator.compute()
    
    print("\n" + "="*40)
    print(f"FINAL RESULTS: {category}")
    print("-" * 40)
    
    # Image Level
    print(f"Image AUROC:   {results['image_auroc']:.4f}")
    print(f"Image F1-Max:  {results['image_f1_max']:.4f}")
    print(f"Image AP:      {results['image_ap']:.4f}")
    print("-" * 40)
    
    # Pixel Level
    if 'pixel_auroc' in results:
        print(f"Pixel AUROC:   {results['pixel_auroc']:.4f}")
        print(f"Pixel F1-Max:  {results['pixel_f1_max']:.4f}")
        #print(f"PRO Score:     {results['pixel_pro']:.4f}") 
    print("="*40)

if __name__ == "__main__":
    main()