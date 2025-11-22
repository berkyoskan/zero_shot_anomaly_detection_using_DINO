# demo_mvtec_simple.py
from __future__ import annotations
from segmenters.sam3 import SAM3Segmenter
import torch
from backbones import get_backbone
from dataset.dataloader import load_mvtec
from models.model_bank_knn import PatchKNNDetector
from utils.visualize import visualize_prediction


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Loading the dataset
    category = "bottle"
    root = "dataset/mvtec_anomaly_detection"
    train_paths, test_paths = load_mvtec(category=category, root=root)
    print(f"{category}: {len(train_paths)} train, {len(test_paths)} test images")


    # Initialize segmenter or keep segmenter = None if you don't want
    segmenter = SAM3Segmenter(text_prompt = category, device=device)

    # Initialize backbone
    backbone = get_backbone("dinov2_small")

    #Initialize model
    model = PatchKNNDetector(
        backbone=backbone,
        segmenter=segmenter,
        device=device,
        k_neighbors=1,
    )

    # identify number of images that will go into memory bank
    n_ref = 1          # one-shot
    # n_ref = 5        # few-shot - havent checked
    model.fit(train_paths, n_ref=n_ref)

    # --- run on a few test images ------------------------------------
    for i, path in enumerate(test_paths[:5]):
        print(f"\n[{i}] Testing on: {path}")
        image, amap, score = model.predict(path)
        visualize_prediction(
            image,
            amap,
            image_score=score,
            title=f"{category} | {path.split('/')[-2:]} | score={score:.3f}",
        )


if __name__ == "__main__":
    main()
