import os
import sys
import time
import datetime
import numpy as np
import albumentations as A
import cv2
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import seeding, create_dir, print_and_save, shuffling, epoch_time, calculate_metrics
from model import TResUnet
from metrics import DiceBCELoss
from fine_tune import DATASET, train, evaluate, load_pretrained_for_finetune, set_trainable


def load_names_combined(path, file_path, mask_dir):
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = [os.path.join(path, "images", name) + ".png" for name in data]
    masks  = [os.path.join(path, mask_dir, name) + ".png" for name in data]
    return images, masks

def load_data_combined(path, mask_dir):
    train_names_path = f"{path}/train.txt"
    valid_names_path = f"{path}/val.txt"

    train_x, train_y = load_names_combined(path, train_names_path, mask_dir)
    valid_x, valid_y = load_names_combined(path, valid_names_path, mask_dir)

    return (train_x, train_y), (valid_x, valid_y)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("p", "vs"):
        print("Usage: python fine_tune_combined.py <p|vs>")
        sys.exit(1)

    task = sys.argv[1]
    mask_dir = "p_masks" if task == "p" else "vs_masks"

    """ Seeding """
    seeding(42)

    """ Directories """
    files_dir = f"ps_p_vs_combined_files/{task}"
    create_dir(files_dir)

    """ Training logfile """
    train_log_path = f"{files_dir}/train_log.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open(train_log_path, "w")
        train_log.write("\n")
        train_log.close()

    """ Record Date & Time """
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    """ Hyperparameters """
    image_size = 256
    size = (image_size, image_size)
    batch_size = 8
    num_epochs = 500
    lr = 1e-5
    early_stopping_patience = 50
    checkpoint_path = f"{files_dir}/checkpoint_finetune.pth"
    path = "ps_p_vs_combined"

    pretrained_path = "liver_pretrained/checkpoint.pth"
    strict_load = True
    finetune_mode = "all"

    data_str = (
        f"Task: {task} (mask_dir: {mask_dir})\n"
        f"Image Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
        f"Early Stopping Patience: {early_stopping_patience}\n"
        f"Pretrained: {pretrained_path}\n"
        f"Strict load: {strict_load}\n"
        f"Finetune mode: {finetune_mode}\n"
    )
    print_and_save(train_log_path, data_str)

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y) = load_data_combined(path, mask_dir)
    train_x, train_y = shuffling(train_x, train_y)
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    """ Data augmentation: Transforms """
    transform = A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    """ Dataset and loader """
    train_dataset = DATASET(train_x, train_y, size, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, size, transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    """ Model """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TResUnet().to(device)

    missing, unexpected = load_pretrained_for_finetune(
        model=model,
        pretrained_path=pretrained_path,
        device=device,
        strict=strict_load
    )
    print_and_save(train_log_path, f"Loaded pretrained weights from: {pretrained_path}")
    if len(missing) > 0:
        print_and_save(train_log_path, f"Missing keys ({len(missing)}): {missing[:20]}{' ...' if len(missing)>20 else ''}")
    if len(unexpected) > 0:
        print_and_save(train_log_path, f"Unexpected keys ({len(unexpected)}): {unexpected[:20]}{' ...' if len(unexpected)>20 else ''}")

    set_trainable(model, finetune_mode=finetune_mode)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5,# verbose=True
    )

    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training the model """
    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_metrics[1] > best_valid_metrics:
            data_str = (
                f"Valid F1 improved from {best_valid_metrics:2.4f} to {valid_metrics[1]:2.4f}. "
                f"Saving checkpoint: {checkpoint_path}"
            )
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[1]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0
        else:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)

        if early_stopping_count >= early_stopping_patience:
            data_str = (
                f"Early stopping: validation F1 did not improve for "
                f"{early_stopping_patience} consecutive epochs.\n"
            )
            print_and_save(train_log_path, data_str)
            break
