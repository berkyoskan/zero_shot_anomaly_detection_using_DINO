
import numpy as np
import matplotlib.pyplot as plt
import cv2

#Completeöy vibecoded
def visualize_prediction(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    image_score: float,
    threshold_percentile: float = 95.0,
    title: str | None = None,
) -> None:
    """
    Show:
        - original image
        - heatmap overlay
        - binary mask overlay (thresholded)
    """
    # Normalize anomaly map to [0, 1] for visualization
    amap = anomaly_map.astype(np.float32)
    amap -= amap.min()
    if amap.max() > 0:
        amap /= amap.max()

    thresh = np.percentile(amap, threshold_percentile)
    binary = amap >= thresh

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(image)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    axes[1].imshow(image)
    im = axes[1].imshow(amap, cmap="jet", alpha=0.5)
    axes[1].set_title("Anomaly heatmap")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(image)
    axes[2].imshow(binary, cmap="gray", alpha=0.5)
    axes[2].set_title(f"Thresholded (>{threshold_percentile:.0f}%)")
    axes[2].axis("off")

    if title is None:
        title = f"Image anomaly score: {image_score:.3f}"

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def visualize_segmentation(
    image: np.ndarray,
    full_mask: np.ndarray,
    grid_size: tuple[int, int] | None = None,
    title: str | None = None,
) -> None:
    """
    Visualize SAM2 segmentation.

    Args:
        image: (H, W, 3) RGB uint8
        full_mask: (H, W) bool or 0/1 array from SAM2
        grid_size: optional (H_patches, W_patches) to also show patch-level mask
        title: optional title string
    """
    img = image
    mask = full_mask.astype(bool)
    H, W = mask.shape

    # Prepare figure layout
    n_cols = 3 if grid_size is None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    # 1) input image
    axes[0].imshow(img)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    # 2) raw binary mask
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("SAM2 mask (full-res)")
    axes[1].axis("off")

    # 3) overlay mask on image
    axes[2].imshow(img)
    axes[2].imshow(mask, cmap="Reds", alpha=0.4)
    axes[2].set_title("Mask overlay")
    axes[2].axis("off")

    # 4) optional patch-level mask (after downsampling)
    if grid_size is not None:
        gh, gw = grid_size
        # downsample full mask to patch grid and back up to image size
        patch_mask_small = cv2.resize(
            mask.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        patch_mask_full = cv2.resize(
            patch_mask_small.astype(np.uint8),
            (W, H),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        axes[3].imshow(img)
        axes[3].imshow(patch_mask_full, cmap="Blues", alpha=0.4)
        axes[3].set_title("Patch-level mask (after downsample)")
        axes[3].axis("off")

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()