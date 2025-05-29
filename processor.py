"""
Inference script for predicting malignancy of lung nodules
"""
import numpy as np
import dataloader
import torch
import torch.nn as nn
from torchvision import models
from models.model_3d import I3D
from models.model_2d import ResNet18
from models.vit_3d import ViT
import os
import math
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

# define processor
class MalignancyProcessor:
    """
    Loads a chest CT scan, and predicts the malignancy around a nodule
    """

    def __init__(self, mode="2D", suppress_logs=False, model_name="LUNA25-baseline-2D"):

        self.size_px = 64
        self.size_mm = 50

        self.model_name = model_name
        self.mode = mode
        self.suppress_logs = suppress_logs

        if not self.suppress_logs:
            logging.info("Initializing the deep learning system")

        if self.mode == "2D":
            self.model_2d = ResNet18(weights=None).cuda()
        elif self.mode == "3D":
            #self.model_3d = I3D(num_classes=1, pre_trained=False, input_channels=3).cuda()
            self.model_3d = ViT(
                image_size=(128, 128),
                image_patch_size=16,
                frames=64,
                frame_patch_size=8,
                dim=1024,
                depth=6,
                heads=8,
                mlp_dim=1024,
                dropout=0.1,
                emb_dropout=0.1,
                num_classes=1
            ).cuda()

        self.model_root = "/opt/app/resources/"

    def define_inputs(self, image, header, coords):
        self.image = image
        self.header = header
        self.coords = coords

    def extract_patch(self, coord, output_shape, mode):

        patch = dataloader.extract_patch(
            CTData=self.image,
            coord=coord,
            srcVoxelOrigin=self.header["origin"],
            srcWorldMatrix=self.header["transform"],
            srcVoxelSpacing=self.header["spacing"],
            output_shape=output_shape,
            voxel_spacing=(
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
            ),
            coord_space_world=True,
            mode=mode,
        )

        # ensure same datatype...
        patch = patch.astype(np.float32)

        # clip and scale...
        patch = dataloader.clip_and_scale(patch)
        return patch

    def _process_model(self, mode, model_name):

        if not self.suppress_logs:
            logging.info("Processing in " + mode)

        if mode == "2D":
            output_shape = [1, self.size_px, self.size_px]
            model = self.model_2d
        else:
            output_shape = [self.size_px, self.size_px, self.size_px]
            model = self.model_3d

        nodules = []

        for _coord in self.coords:

            patch = self.extract_patch(_coord, output_shape, mode=mode)
            nodules.append(patch)

        nodules = np.array(nodules)
        nodules = torch.from_numpy(nodules).cuda()

        ckpt = torch.load(
            os.path.join(
                self.model_root,
                model_name,
                "best_metric_model.pth",
            ),
            map_location="cuda:0"
        )
        model.load_state_dict(ckpt)
        model.eval()
        logits = model(nodules)
        logits = logits.data.cpu().numpy()

        logits = np.array(logits)
        return logits
    
    def predict(self):
        if isinstance(self.model_name, list):
            logits_list = []
            for model_name in self.model_name:
                logits = self._process_model(self.mode, model_name) 
                logits_list.append(logits)

            # Stack logits from each model: shape (num_models, num_samples, num_classes)
            stacked_logits = np.stack(logits_list, axis=0)

            # Average logits across models: shape (num_samples, num_classes)
            avg_logits = np.mean(stacked_logits, axis=0)
        else:
            avg_logits = self._process_model(self.mode, self.model_name)

        # Convert to probability
        probability = torch.sigmoid(torch.from_numpy(avg_logits)).numpy()
        return probability, avg_logits
