from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
import numpy as np
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from PIL import Image
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Fastflow
from anomalib.utils.post_processing import superimpose_anomaly_map
import timm


pre_processor = Patchcore.configure_pre_processor(image_size=(224, 224))


datamodule = MVTecAD(
    root="/home/adam/Desktop/datasets/MVTecAD/archive",
    category="bottle", 
    num_workers=0,
)


backbone = timm.create_model(
    "vit_small_patch16_224.dino",
    pretrained=True,
    features_only=True,
)


backbone_name = "vit_small_patch16_224.dino"

model = Patchcore(
    backbone=backbone_name,          
    layers=("blocks.10", "blocks.11"),  
    coreset_sampling_ratio=0.1,
    pre_processor = pre_processor
)


engine = Engine(
    accelerator="auto",  
    logger=False,
)

engine.fit(datamodule=datamodule, model=model)


engine.test(datamodule=datamodule, model=model)


data_path = "/home/adam/Desktop/datasets/MVTecAD/archive/bottle/test/broken_large/000.png"
predictions = engine.predict(model=model, data_path=data_path)
prediction = predictions[0]  # Get the first and only prediction

print(
    f"Image Shape: {prediction.image.shape},\n"
    f"Anomaly Map Shape: {prediction.anomaly_map.shape}, \n"
    f"Predicted Mask Shape: {prediction.pred_mask.shape}",
)



image_path = prediction.image_path[0]
image_size = prediction.image.shape[-2:]        


image = np.array(Image.open(image_path).convert("RGB").resize(image_size))


anomaly_map = prediction.anomaly_map[0].cpu().numpy()
anomaly_map = np.squeeze(anomaly_map)         

overlay = superimpose_anomaly_map(
    anomaly_map=anomaly_map,
    image=image,
    alpha=0.5
)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].imshow(image)
axs[0].set_title("Original")
axs[0].axis("off")

axs[1].imshow(anomaly_map, cmap="jet")
axs[1].set_title("Anomaly Map")
axs[1].axis("off")

axs[2].imshow(overlay)
axs[2].set_title("Overlay")
axs[2].axis("off")

plt.tight_layout()
plt.show()

