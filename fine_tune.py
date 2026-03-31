import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import cv2
from PIL import Image
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import seeding, create_dir, print_and_save, shuffling, epoch_time, calculate_metrics
from model import TResUnet
from metrics import DiceLoss, DiceBCELoss

def load_names(path, file_path):
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = [os.path.join(path, "images", name) + ".png" for name in data]
    masks  = [os.path.join(path, "masks",  name) + ".png" for name in data]
    return images, masks

def load_data(path):
    train_names_path = f"{path}/train.txt"
    valid_names_path = f"{path}/val.txt"

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)

    return (train_x, train_y), (valid_x, valid_y)

class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        self.size = size  # <- keep size here

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask  = cv2.imread(self.masks_path[index],  cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask  = augmentations["mask"]

        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0

        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0

        return image, mask

    def __len__(self):
        return self.n_samples

def train(model, loader, optimizer, loss_fn, device):
    model.train()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        batch_jac, batch_f1, batch_recall, batch_precision = [], [], [], []
        for yt, yp in zip(y, y_pred):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    epoch_loss /= len(loader)
    epoch_jac /= len(loader)
    epoch_f1 /= len(loader)
    epoch_recall /= len(loader)
    epoch_precision /= len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]

def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            batch_jac, batch_f1, batch_recall, batch_precision = [], [], [], []
            for yt, yp in zip(y, y_pred):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

    epoch_loss /= len(loader)
    epoch_jac /= len(loader)
    epoch_f1 /= len(loader)
    epoch_recall /= len(loader)
    epoch_precision /= len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]

def load_pretrained_for_finetune(model: nn.Module, pretrained_path: str, device: torch.device, strict: bool = True):
    """
    Loads weights into `model` from `pretrained_path`.
    Works for either:
      - a plain state_dict saved via torch.save(model.state_dict(), ...)
      - a checkpoint dict that contains 'state_dict' or 'model_state_dict'
    """
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

    ckpt = torch.load(pretrained_path, map_location=device)

    # Support a few common checkpoint formats
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            # might already be a raw state_dict-like dict
            state = ckpt
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=strict)
    return missing, unexpected

def set_trainable(model: nn.Module, finetune_mode: str):
    """
    finetune_mode:
      - "all": train everything
      - "decoder": freeze encoder, train decoder (best effort, depends on your module names)
      - "last": only train last layers (best effort)
    """
    finetune_mode = finetune_mode.lower()

    # default: unfreeze all
    for p in model.parameters():
        p.requires_grad = True

    if finetune_mode == "all":
        return

    # Best-effort freezing based on common naming patterns.
    # Adjust these keys to match your TResUnet implementation if needed.
    if finetune_mode == "decoder":
        freeze_keywords = ["encoder", "backbone", "stem"]
    elif finetune_mode == "last":
        freeze_keywords = ["encoder", "backbone", "stem", "bottleneck"]
    else:
        raise ValueError(f"Unknown finetune_mode: {finetune_mode}. Use one of: all|decoder|last")

    for name, p in model.named_parameters():
        if any(k in name.lower() for k in freeze_keywords):
            p.requires_grad = False

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Training logfile """
    train_log_path = "files/train_log.txt"
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

    # Fine-tuning typically uses a lower LR than training from scratch
    lr = 1e-5

    early_stopping_patience = 50
    checkpoint_path = "files/checkpoint_finetune.pth"
    path = "ps_vs_new"

    # >>> NEW: path to pretrained weights you want to fine-tune from
    pretrained_path = "liver_pretrained/checkpoint.pth"   # change this to your pretrained checkpoint
    strict_load = True                         # set False if you changed heads / layer names
    finetune_mode = "all"                      # "all" | "decoder" | "last"

    data_str = (
        f"Image Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
        f"Early Stopping Patience: {early_stopping_patience}\n"
        f"Pretrained: {pretrained_path}\n"
        f"Strict load: {strict_load}\n"
        f"Finetune mode: {finetune_mode}\n"
    )
    print_and_save(train_log_path, data_str)

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y) = load_data(path)
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

    # >>> NEW: load pretrained weights
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

    # >>> NEW: choose what to fine-tune
    set_trainable(model, finetune_mode=finetune_mode)

    # Build optimizer only over trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, verbose=True
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