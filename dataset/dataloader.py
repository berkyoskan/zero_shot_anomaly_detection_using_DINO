import os
from typing import List, Tuple

from anomalib.data import MVTec, MVTecAD
from anomalib.data.datasets.image import MVTecDataset
"""Loading logic :
   Unfortrunately you need to download MVtech yourself and place in dataset folder

"""


def load_mvtec(
    category: str,
    root: str = "./datasets/MVTec",
) -> Tuple[list[str], list[str]]:


    train_ds = MVTecDataset(
        root=root,
        category=category,
        split="train"
    )

    test_ds = MVTecDataset(
        root=root,
        category=category,
        split="test"
    )

    train_paths = train_ds.samples["image_path"].tolist()
    test_paths = test_ds.samples["image_path"].tolist()
    return train_paths, test_paths
