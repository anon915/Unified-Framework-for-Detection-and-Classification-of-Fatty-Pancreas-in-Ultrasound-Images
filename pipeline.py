
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings("ignore")

import csv
import time
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import TResUnet
from utils import seeding, create_dir

# --- Config ---
DATA_PATH = "ps_p_vs_combined"
FILES_DIR = "ps_p_vs_combined_files"
LABELS_CSV = f"{DATA_PATH}/labels.csv"
SAVE_DIR = "results/pipeline"

P_CHECKPOINT = f"{FILES_DIR}/p/checkpoint_finetune.pth"
VS_CHECKPOINT = f"{FILES_DIR}/vs/checkpoint_finetune.pth"

SIZE = (256, 256)
PATCH_SIZE = (3, 3)
DELTA = 20
HIST_BINS = 32


# ============================================================
# STAGE 1: SEGMENTATION
# ============================================================

def load_segmentation_models(p_checkpoint, vs_checkpoint, device):
    p_model = TResUnet().to(device)
    p_model.load_state_dict(torch.load(p_checkpoint, map_location=device))
    p_model.eval()

    vs_model = TResUnet().to(device)
    vs_model.load_state_dict(torch.load(vs_checkpoint, map_location=device))
    vs_model.eval()

    return p_model, vs_model


def predict_mask(model, image, device):
    x = np.transpose(image, (2, 0, 1)) / 255.0
    x = np.expand_dims(x, axis=0).astype(np.float32)
    x = torch.from_numpy(x).to(device)

    with torch.no_grad():
        pred = model(x)
        pred = torch.sigmoid(pred)
        mask = (pred[0].cpu().numpy().squeeze() > 0.5).astype(np.uint8) * 255

    return mask


# ============================================================
# STAGE 2: ANATOMICALLY-GUIDED PATCH EXTRACTION
# ============================================================

def extract_pancreas_patches(p_mask, vs_mask, ultrasound, patch_size):
    ys, xs = np.where(p_mask > 127)
    if len(ys) == 0:
        return None

    H, W = ultrasound.shape[:2]
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    patches = []
    for py in range(y_min, y_max + 1, patch_size[0]):
        for px in range(x_min, x_max + 1, patch_size[1]):
            if py + patch_size[0] > H or px + patch_size[1] > W:
                continue
            if not np.all(p_mask[py:py+patch_size[0], px:px+patch_size[1]] > 127):
                continue
            if np.any(vs_mask[py:py+patch_size[0], px:px+patch_size[1]] > 127):
                continue
            patches.append(ultrasound[py:py+patch_size[0], px:px+patch_size[1]])

    return np.array(patches) if patches else None


def extract_fat_patches(p_mask, vs_mask, ultrasound, patch_size, delta):
    ys, xs = np.where(vs_mask > 127)
    if len(ys) == 0:
        return None

    H, W = ultrasound.shape[:2]
    x_min, x_max = xs.min(), xs.max()

    # Bottom contour of the splenic vein
    bottom_contour = {}
    for x in range(x_min, x_max + 1):
        col_ys = ys[xs == x]
        if len(col_ys) > 0:
            bottom_contour[x] = col_ys.max()

    if not bottom_contour:
        return None

    # Valid extraction zone: delta pixels below the vein contour
    valid_mask = np.zeros((H, W), dtype=bool)
    for x, y_bottom in bottom_contour.items():
        y_start = y_bottom + 1
        y_end = min(H, y_bottom + 1 + delta)
        valid_mask[y_start:y_end, x] = True

    valid_mask[vs_mask > 127] = False
    valid_mask[p_mask > 127] = False

    valid_ys, valid_xs = np.where(valid_mask)
    if len(valid_ys) == 0:
        return None

    patches = []
    for py in range(min(valid_ys), max(valid_ys) + 1, patch_size[0]):
        for px in range(min(valid_xs), max(valid_xs) + 1, patch_size[1]):
            if py + patch_size[0] > H or px + patch_size[1] > W:
                continue
            if not valid_mask[py:py+patch_size[0], px:px+patch_size[1]].all():
                continue
            patches.append(ultrasound[py:py+patch_size[0], px:px+patch_size[1]])

    return np.array(patches) if patches else None


# ============================================================
# STAGE 3: FEATURE ENGINEERING + CLASSIFICATION
# ============================================================

def patch_feature_vector(patch, hist_bins):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float64)

    mean = np.mean(gray)
    std = np.std(gray)
    median = np.median(gray)
    hist, _ = np.histogram(gray.flatten(), bins=hist_bins, range=(0, 255), density=True)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = np.var(laplacian)

    local_mean = cv2.blur(gray, (3, 3))
    local_sq_mean = cv2.blur(gray ** 2, (3, 3))
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
    local_contrast = np.mean(np.sqrt(local_var))

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mean = np.mean(np.sqrt(gx ** 2 + gy ** 2))

    return np.concatenate([
        [mean, std, median, lap_var, local_contrast, grad_mean],
        hist,
    ])


def extract_patient_features(fat_patches, pancreas_patches, hist_bins):
    fat_feats = np.array([patch_feature_vector(p, hist_bins) for p in fat_patches])
    panc_feats = np.array([patch_feature_vector(p, hist_bins) for p in pancreas_patches])

    # Normalize jointly
    all_feats = np.vstack([fat_feats, panc_feats])
    mu = all_feats.mean(axis=0)
    sigma = all_feats.std(axis=0)
    sigma[sigma < 1e-10] = 1.0

    fat_norm = (fat_feats - mu) / sigma
    panc_norm = (panc_feats - mu) / sigma

    # Pairwise L2 distance matrix
    diff = fat_norm[:, np.newaxis, :] - panc_norm[np.newaxis, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))
    all_dists = dist_matrix.flatten()

    # Distance distribution statistics
    min_per_fat = dist_matrix.min(axis=1)
    mean_feat_diff = fat_feats.mean(axis=0) - panc_feats.mean(axis=0)

    features = np.concatenate([
        [np.mean(all_dists), np.std(all_dists), np.median(all_dists),
         np.percentile(all_dists, 10), np.percentile(all_dists, 90)],
        [np.mean(min_per_fat), np.std(min_per_fat),
         np.mean(all_dists < np.percentile(all_dists, 25))],
        mean_feat_diff,
    ])

    return features


def classify_patients(X, n_clusters=2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Identify fatty cluster: lower mean pairwise distance = more similar to fat
    dist_means = X[:, 0]
    c0 = dist_means[labels == 0].mean() if (labels == 0).any() else float("inf")
    c1 = dist_means[labels == 1].mean() if (labels == 1).any() else float("inf")
    fatty_cluster = 0 if c0 < c1 else 1

    return labels, fatty_cluster, kmeans, X_scaled


# ============================================================
# VISUALIZATION
# ============================================================

def plot_pca(X_scaled, labels, fatty_cluster, patient_names, save_dir):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    for c, (label, color, marker) in enumerate([
        ("Healthy", "#2ecc71", "o"), ("Fatty", "#e74c3c", "^")
    ]):
        cluster_id = fatty_cluster if label == "Fatty" else (1 - fatty_cluster)
        mask = labels == cluster_id
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label,
                   alpha=0.7, s=50, edgecolors="k", linewidth=0.5, marker=marker)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Patient Clusters (PCA)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "patient_clusters_pca.png"), dpi=150)
    plt.close()


# ============================================================
# PIPELINE
# ============================================================

def load_gt_labels(csv_path):
    if not os.path.exists(csv_path):
        return None
    gt = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[row["image_name"]] = row["label"]
    return gt


def run_pipeline():
    seeding(42)
    create_dir(SAVE_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    """ Stage 1: Load segmentation models """
    t0 = time.time()
    p_model, vs_model = load_segmentation_models(P_CHECKPOINT, VS_CHECKPOINT, device)
    t1 = time.time()

    """ Stage 2+3: Extract patches and features per patient """
    image_paths = sorted(glob(os.path.join(DATA_PATH, "images", "*.png")))
    print(f"Found {len(image_paths)} images")

    patient_names = []
    patient_features = []

    for img_path in tqdm(image_paths, desc="Processing"):
        name = os.path.splitext(os.path.basename(img_path))[0]
        us = cv2.resize(cv2.imread(img_path, cv2.IMREAD_COLOR), SIZE)

        p_mask = predict_mask(p_model, us, device)
        vs_mask = predict_mask(vs_model, us, device)

        fat_patches = extract_fat_patches(p_mask, vs_mask, us, PATCH_SIZE, DELTA)
        panc_patches = extract_pancreas_patches(p_mask, vs_mask, us, PATCH_SIZE)

        if fat_patches is None or panc_patches is None:
            continue

        features = extract_patient_features(fat_patches, panc_patches, HIST_BINS)
        patient_names.append(name)
        patient_features.append(features)

    t2 = time.time()

    X = np.array(patient_features)

    """ Classification """
    labels, fatty_cluster, kmeans, X_scaled = classify_patients(X)
    t3 = time.time()

    n_fatty = (labels == fatty_cluster).sum()
    n_healthy = (labels != fatty_cluster).sum()
    print(f"Fatty: {n_fatty}, Healthy: {n_healthy}")

    for name, label in zip(patient_names, labels):
        tag = "FATTY" if label == fatty_cluster else "HEALTHY"
        print(f"  {name}: {tag}")

    """ Evaluate against ground truth if available """
    gt_labels = load_gt_labels(LABELS_CSV)
    if gt_labels:
        from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

        y_pred = np.array([1 if l == fatty_cluster else 0 for l in labels])
        common = [i for i, n in enumerate(patient_names) if n in gt_labels]
        y_true = np.array([1 if gt_labels[patient_names[i]] == "fatty" else 0 for i in common])
        y_pred_common = y_pred[common]

        # Check if flipped assignment is better
        if accuracy_score(y_true, 1 - y_pred_common) > accuracy_score(y_true, y_pred_common):
            y_pred_common = 1 - y_pred_common

        acc = accuracy_score(y_true, y_pred_common)
        f1 = f1_score(y_true, y_pred_common)
        kappa = cohen_kappa_score(y_true, y_pred_common)
        print(f"\nEvaluation ({len(common)} labeled patients): "
              f"Acc={acc:.3f} F1={f1:.3f} Kappa={kappa:.3f}")

    """ Visualization """
    plot_pca(X_scaled, labels, fatty_cluster, patient_names, SAVE_DIR)

    """ Timing """
    print(f"\nWall clock times:")
    print(f"  Model loading:                     {t1-t0:.2f}s")
    print(f"  Segmentation + feature extraction:  {t2-t1:.2f}s ({(t2-t1)/len(image_paths):.3f}s/image)")
    print(f"  Classification:                     {t3-t2:.2f}s")
    print(f"  Total:                              {t3-t0:.2f}s")
    print(f"Results saved to {SAVE_DIR}/")


if __name__ == "__main__":
    run_pipeline()
