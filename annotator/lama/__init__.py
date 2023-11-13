# https://github.com/advimman/lama
import sys

import yaml
import torch
from omegaconf import OmegaConf
import numpy as np
import urllib.request

from einops import rearrange
import os
percorso_radice_progetto = os.path.abspath(os.path.dirname(__file__))
from .saicinpainting.training.trainers import load_checkpoint

# Aggiungi il percorso alla variabile d'ambiente PYTHONPATH
sys.path.append(percorso_radice_progetto)


devices = 'cuda' if torch.cuda.is_available() else 'cpu'
models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models','lama')
os.makedirs(models_path, exist_ok=True)

class LamaInpainting:
    model_dir = os.path.join(models_path, "lama")

    def __init__(self):
        self.model = None
        self.device = devices

    

    def _load_file_from_url(model_path: str, model_dir: str) -> None:
        os.makedirs(os.path.dirname(model_dir), exist_ok=True)
        urllib.request.urlretrieve(model_path, os.path.join(model_dir, model_path.split("/")[-1]))
    
    def load_model(self):
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetLama.pth"
        modelpath = os.path.join(self.model_dir, "ControlNetLama.pth")
        if not os.path.exists(modelpath):
            _load_file_from_url(remote_model_path, model_dir=self.model_dir)
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
        cfg = yaml.safe_load(open(config_path, 'rt'))
        cfg = OmegaConf.create(cfg)
        cfg.training_model.predict_only = True
        cfg.visualizer.kind = 'noop'
        self.model = load_checkpoint(cfg, os.path.abspath(modelpath), strict=False, map_location='cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def unload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, input_image):
        from PIL import Image
        if self.model is None:
            self.load_model()
        self.model.to(self.device)
        color = np.ascontiguousarray(input_image[:, :, 0:3]).astype(np.float32) / 255

        mask = np.ascontiguousarray(input_image[:, :, 3:4]).astype(np.float32) / 255

        with torch.no_grad():
            color = torch.from_numpy(color).float().to(self.device)
            mask = torch.from_numpy(mask).float().to(self.device)
            mask = (mask > 0.5).float()
            color = color * (1 - mask)
            image_feed = torch.cat([color, mask], dim=2)

            image_feed = rearrange(image_feed, 'h w c -> 1 c h w')
            result = self.model(image_feed)[0]
            result = rearrange(result, 'c h w -> h w c')
            result = result * mask + color * (1 - mask)
            result *= 255.0
            return result.detach().cpu().numpy().clip(0, 255).astype(np.uint8)
