import torch
import torchvision
import lightning.pytorch as pl
import medmnist
import lifelines

print("torch version: {}".format(torch.__version__))
print("torchvision version: {}".format(torchvision.__version__))
print("pytorch_lightning version: {}".format(pl.__version__))
print("medmnist version: {}".format(medmnist.__version__))
print("lifelines version: {}".format(lifelines.__version__))

print("Successfully loaded packages")