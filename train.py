"""
Script for training a ResNet18 or I3D to classify a pulmonary nodule as benign or malignant.
"""
from models.model_2d import ResNet18
from models.model_3d import I3D
from dataloader import get_data_loader
import logging
import numpy as np
import torch
import sklearn.metrics as metrics
from tqdm import tqdm
import warnings
import random
import pandas as pd
from experiment_config import config
from datetime import datetime
import argparse
import combining_loss
from models.res_net import ResNet3D_MC3
from models.vit_3d import ViT
from focal_loss import FocalLoss

from sklearn.model_selection import StratifiedKFold
from pathlib import Path


torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

def make_weights_for_balanced_classes(labels):
    """Making sampling weights for the data samples
    :returns: sampling weights for dealing with class imbalance problem

    """
    n_samples = len(labels)
    unique, cnts = np.unique(labels, return_counts=True)
    cnt_dict = dict(zip(unique, cnts))

    weights = []
    for label in labels:
        weights.append(n_samples / float(cnt_dict[label]))
    return weights


def train(
    train_csv_path,
    valid_csv_path,
    exp_save_root,

):
    """
    Train a ResNet18 or an I3D model
    """
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    logging.info(f"Training with {train_csv_path}")
    logging.info(f"Validating with {valid_csv_path}")

    train_df = pd.read_csv(train_csv_path)
    valid_df = pd.read_csv(valid_csv_path)

    print()

    logging.info(
        f"Number of malignant training samples: {train_df.label.sum()}"
    )
    logging.info(
        f"Number of benign training samples: {len(train_df) - train_df.label.sum()}"
    )
    print()
    logging.info(
        f"Number of malignant validation samples: {valid_df.label.sum()}"
    )
    logging.info(
        f"Number of benign validation samples: {len(valid_df) - valid_df.label.sum()}"
    )

    # create a training data loader
    weights = make_weights_for_balanced_classes(train_df.label.values)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_df))

    train_loader = get_data_loader(
        config.DATADIR,
        train_df,
        mode=config.MODE,
        sampler=sampler,
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=config.ROTATION,
        translations=config.TRANSLATION,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
    )

    valid_loader = get_data_loader(
        config.DATADIR,
        valid_df,
        mode=config.MODE,
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=None,
        translations=None,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
    )

    device = torch.device("cuda:0")

    if config.MODEL == "2D":
        model = ResNet18().to(device)
    elif config.MODEL == "3D":
        model = I3D(
            num_classes=1,
            input_channels=3,
            pre_trained=True,
            freeze_bn=True,
        ).to(device)
    elif config.MODEL == "ResNet3D_MC3":
        model = ResNet3D_MC3(
            num_classes=1,
            pretrained=True,
        ).to(device)
    elif config.MODEL == "vit":
        model = ViT(
            image_size=config.VIT["image_size"], # image size
            frames=config.VIT["frames"], # number of frames
            image_patch_size=config.VIT["image_patch_size"],     # image patch size
            frame_patch_size=config.VIT["frame_patch_size"],      # frame patch size
            num_classes=1,
            dim=config.VIT["dim"],
            depth=config.VIT["depth"],
            heads=config.VIT["heads"],
            mlp_dim=config.VIT["mlp_dim"],
            dropout=config.VIT["dropout"],
            emb_dropout=config.VIT["emb_dropout"]
        ).to(device)
        print(config)
    else:
        raise ValueError(f"Unknown model {config.MODEL}")

    if config.LOSS == "BCE":
        loss_function = torch.nn.BCEWithLogitsLoss()
    elif config.LOSS == "BCE_pos_weight":
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.POS_WEIGHT], device=device))
    elif config.LOSS == "Focal":
        loss_function = FocalLoss(alpha=0.25, gamma=2.0).to(device)
    elif config.LOSS == "Combo":
        loss_function = combining_loss.ComboLoss(alpha=0.3, gamma=2.0, dice_weight=0.0).to(device)
    else:
        raise ValueError(f"Unknown loss function {config.LOSS}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    epochs = config.EPOCHS
    patience = config.PATIENCE
    counter = 0

    for epoch in range(epochs):

        if counter > patience:
            logging.info(f"Model not improving for {patience} epochs")
            break

        logging.info("-" * 10)
        logging.info("epoch {}/{}".format(epoch + 1, epochs))

        # train

        model.train()

        epoch_loss = 0
        step = 0

        for batch_data in tqdm(train_loader):
            step += 1
            inputs, labels = batch_data["image"], batch_data["label"]
            labels = labels.float().to(device)
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_df) // train_loader.batch_size
            if step % 100 == 0:
                logging.info(
                    "{}/{}, train_loss: {:.4f}".format(step, epoch_len, loss.item())
                )
        epoch_loss /= step
        logging.info(
            "epoch {} average train loss: {:.4f}".format(epoch + 1, epoch_loss)
        )

        # validate

        model.eval()

        epoch_loss = 0
        step = 0

        with torch.no_grad():

            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.float32, device=device)
            for val_data in valid_loader:
                step += 1
                val_images, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_images = val_images.to(device)
                val_labels = val_labels.float().to(device)
                outputs = model(val_images)
                loss = loss_function(outputs.squeeze(), val_labels.squeeze())
                epoch_loss += loss.item()
                y_pred = torch.cat([y_pred, outputs], dim=0)
                y = torch.cat([y, val_labels], dim=0)

                epoch_len = len(valid_df) // valid_loader.batch_size

            epoch_loss /= step
            logging.info(
                "epoch {} average valid loss: {:.4f}".format(epoch + 1, epoch_loss)
            )

            y_pred = torch.sigmoid(y_pred.reshape(-1)).data.cpu().numpy().reshape(-1)
            y = y.data.cpu().numpy().reshape(-1)

            fpr, tpr, _ = metrics.roc_curve(y, y_pred)
            auc_metric = metrics.auc(fpr, tpr)

            if auc_metric > best_metric:

                counter = 0
                best_metric = auc_metric
                best_metric_epoch = epoch + 1

                torch.save(
                    model.state_dict(),
                    exp_save_root / "best_metric_model.pth",
                )

                metadata = {
                    "train_csv": train_csv_path,
                    "valid_csv": valid_csv_path,
                    "config": config,
                    "best_auc": best_metric,
                    "epoch": best_metric_epoch,
                }
                np.save(
                    exp_save_root / "config.npy",
                    metadata,
                )

                logging.info("saved new best metric model")

            logging.info(
                "current epoch: {} current AUC: {:.4f} best AUC: {:.4f} at epoch {}".format(
                    epoch + 1, auc_metric, best_metric, best_metric_epoch
                )
            )
        counter += 1

    logging.info(
        "train completed, best_metric: {:.4f} at epoch: {}".format(
            best_metric, best_metric_epoch
        )
    )


if __name__ == "__main__":

    if config.CSV_DIR_VALID is not None:
        experiment_name = f"{config.EXPERIMENT_NAME}-{config.MODE}-{datetime.today().strftime('%Y%m%d')}"
        exp_save_root = config.EXPERIMENT_DIR / experiment_name
        exp_save_root.mkdir(parents=True, exist_ok=True)

        # Standard training run
        train(
            train_csv_path=config.CSV_DIR_TRAIN,
            valid_csv_path=config.CSV_DIR_VALID,
            exp_save_root=exp_save_root,
        )

    else:
        # Load full dataset
        df = pd.read_csv(config.CSV_DIR_TRAIN)

        # Create patient-level summary with single label per patient
        patient_labels = df.groupby("PatientID")["label"].max().reset_index()

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=config.SEED)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_labels["PatientID"], patient_labels["label"])):
            train_patients = patient_labels.iloc[train_idx]
            val_patients = patient_labels.iloc[val_idx]

            train_df = df[df['PatientID'].isin(train_patients['PatientID'])]
            val_df = df[df['PatientID'].isin(val_patients['PatientID'])]

            # Create fold-specific experiment dir
            fold_name = f"{config.EXPERIMENT_NAME}-{config.MODE}-fold{fold_idx}-{datetime.today().strftime('%Y%m%d')}"
            fold_exp_dir = config.EXPERIMENT_DIR / fold_name
            Path(fold_exp_dir).mkdir(parents=True, exist_ok=True)

            # Save CSVs
            train_csv_path = fold_exp_dir / "train.csv"
            val_csv_path = fold_exp_dir / "valid.csv"
            train_df.to_csv(train_csv_path, index=False)
            val_df.to_csv(val_csv_path, index=False)

            print(f"Starting training for fold {fold_idx}")

            train(
                train_csv_path=train_csv_path,
                valid_csv_path=val_csv_path,
                exp_save_root=fold_exp_dir,
            )
