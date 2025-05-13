import torch
torch.hub.set_dir('/home/phd_li/.cache/torch/hub')
import torchinfo
import loralib as lora
# from models.dino.hubconf import dino_vits8
from dinov1_models import dino_vits8
# This sets requires_grad to False for all parameters without the string "lora_" in their names


# vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
vits8 = dino_vits8()

lora.mark_only_lora_as_trainable(vits8)

# torchinfo.summary(vits8, depth=5)