import numpy as np
import cv2
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from skimage.measure import label, regionprops

class AnomalyEvaluator:
    def __init__(self, pixel_subsample_rate=0.01, compute_pro=False):
        """
        Args:
            pixel_subsample_rate: Fraction of pixels to keep for pixel-metrics 
                                  (only used if compute_pro=False).
            compute_pro: If True, stores FULL resolution maps to calculate 
                         Per-Region Overlap (uses more RAM).
        """
        self.subsample_rate = pixel_subsample_rate
        self.compute_pro = compute_pro
        self.reset()

    def reset(self):
        # Image-level storage
        self.img_preds = []
        self.img_labels = []
        
        # Pixel-level storage
        self.pix_preds = [] # Flat list of pixels (for AUROC)
        self.pix_labels = [] # Flat list of labels (for AUROC)
        
        # Full Map storage (Only used if compute_pro=True)
        self.full_amaps = []
        self.full_masks = []

    def update(self, image_score, gt_label, anomaly_map=None, gt_mask=None):
        """
        Call this inside your loop for every image.
        """
        # 1. Store Image Level Data
        self.img_preds.append(image_score)
        self.img_labels.append(gt_label)

        # 2. Store Pixel Level Data
        if anomaly_map is not None and gt_mask is not None:
            self._update_pixel_metrics(anomaly_map, gt_mask)

    def _update_pixel_metrics(self, amap, mask):
        # Resize mask if needed
        if mask.shape != amap.shape:
            mask = cv2.resize(mask, (amap.shape[1], amap.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        mask = (mask > 0).astype(int)

        # A. If calculating PRO, we must store the FULL maps
        if self.compute_pro:
            self.full_amaps.append(amap)
            self.full_masks.append(mask)

        # B. For standard AUROC/F1, we can just store flattened pixels
        # (If PRO is on, we flatten the full map. If off, we subsample)
        flat_amap = amap.flatten()
        flat_mask = mask.flatten()

        if self.compute_pro or self.subsample_rate >= 1.0:
            self.pix_preds.extend(flat_amap)
            self.pix_labels.extend(flat_mask)
        else:
            # Random Subsampling to save memory
            num_pixels = len(flat_mask)
            sample_size = int(num_pixels * self.subsample_rate)
            indices = np.random.choice(num_pixels, sample_size, replace=False)
            self.pix_preds.extend(flat_amap[indices])
            self.pix_labels.extend(flat_mask[indices])

    def compute(self):
        results = {}
        y_true = np.array(self.img_labels)
        y_score = np.array(self.img_preds)
        
        # --- 1. Image-Level AUROC ---
        results['image_auroc'] = roc_auc_score(y_true, y_score)
        
        # --- 2. Image-Level AP (Average Precision) ---
        results['image_ap'] = average_precision_score(y_true, y_score)

        # --- 3. Image-Level F1-max ---
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        results['image_f1_max'] = np.max(f1_scores)

        # --- Pixel Metrics Setup ---
        if len(self.pix_labels) > 0:
            pix_true = np.array(self.pix_labels)
            pix_score = np.array(self.pix_preds)
            
            # --- 4. Pixel-Level AUROC ---
            results['pixel_auroc'] = roc_auc_score(pix_true, pix_score)
            
            # Find optimal pixel threshold based on F1-max
            # We need this for the PRO calculation
            prec_p, rec_p, thresholds_p = precision_recall_curve(pix_true, pix_score)
            f1_p = 2 * (prec_p * rec_p) / (prec_p + rec_p + 1e-8)
            best_idx = np.argmax(f1_p)
            best_threshold = thresholds_p[best_idx] if best_idx < len(thresholds_p) else 0.5
            
            results['pixel_f1_max'] = np.max(f1_p)

            # --- 5. Per-Region Overlap (PRO) ---
            if self.compute_pro:
                results['pixel_pro'] = self._compute_pro(best_threshold)

        return results

    def _compute_pro(self, threshold):
        """
        Calculates PRO at the specific optimal threshold.
        PRO = Average coverage of connected ground truth components.
        """
        total_pro = 0
        n_defects = 0

        for i in range(len(self.full_amaps)):
            gt = self.full_masks[i]
            # Skip normal images (no defects to overlap)
            if np.sum(gt) == 0:
                continue

            pred_mask = (self.full_amaps[i] >= threshold).astype(int)

            # Label connected components in Ground Truth
            labeled_gt = label(gt)
            regions = regionprops(labeled_gt)

            for region in regions:
                n_defects += 1
                # Create a mask for just this one defect blob
                blob_mask = (labeled_gt == region.label)
                
                # How many pixels of this blob did we find?
                overlap_pixels = np.sum(pred_mask & blob_mask)
                blob_area = region.area
                
                # Add coverage fraction (0.0 to 1.0)
                total_pro += (overlap_pixels / blob_area)

        return total_pro / n_defects if n_defects > 0 else 0.0