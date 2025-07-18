import os
import argparse
from pathlib import Path
from typing import List, Tuple
from test_dataset import TestDataset
from tqdm import tqdm
from utils import find_optimal_threshold

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score,recall_score
from dataset import ECGDataset
from utils import split_data
from vHeat import vHeat1D

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate vHeat model")
    parser.add_argument("--data-dir", type=str, default="test_data",
                        help="Path to data folder containing .hea and .dat files")
    parser.add_argument("--model-path", type=str,default="model/vHeat_base_data_all_42_80.pth",
                        help="Path to the saved model .pth file")
    parser.add_argument("--output-dir", type=str,default="outputs",
                        help="Directory to save prediction text files")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for DataLoader")
    return parser.parse_args()

def build_dataloader(data_dir, batch_size=64):
    dataset = TestDataset(data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    model = vHeat1D().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def infer(model, loader, device) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    all_probs, all_labels, all_pids = [], [], []

    for data, labels, pids in tqdm(loader):
        data = data.to(device)
        probs = torch.sigmoid(model(data)).squeeze(1)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
        all_pids.extend(pids)

    return (
        torch.cat(all_probs).numpy(),
        torch.cat(all_labels).numpy(),
        all_pids,
    )

def save_predictions(probs, labels, pids, threshold, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for prob, label, pid in tqdm(zip(probs, labels, pids), total=len(probs), desc="Processing"):
        pred = int(prob >= threshold)
        content = (
            f"{pid}\n"
            f"# Chagas label: {pred}\n"
            f"# Chagas probability: {prob:.6f}\n"
        )
        with open(os.path.join(output_dir, f"{pid}.txt"), "w", encoding="utf-8") as f:
            f.write(content)

def calc_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "ACC": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
    }

def get_thresholds(val_loader, net, device):
    print('Finding optimal thresholds...')
    output_list, label_list = [], []

    for data, labels, _ in tqdm(val_loader, desc="Threshold Calculation"):
        data, labels = data.to(device), labels.to(device)
        output = net(data)
        output = torch.sigmoid(output).squeeze()
        output_list.append(output.detach().cpu().numpy())
        label_list.append(labels.detach().cpu().numpy())

    y_trues = np.concatenate(label_list, axis=0)
    y_scores = np.concatenate(output_list, axis=0)

    threshold = find_optimal_threshold(y_trues, y_scores)

    return threshold

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.model_path, device)

    data_dir = 'data'
    label_csv = 'data/label.csv'

    leads = 'all'
    train_folds, val_folds, test_folds = split_data(seed=42)
    val_dataset = ECGDataset('val', data_dir, label_csv, val_folds, leads)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                            pin_memory=True)
    threshold = get_thresholds(val_loader, model, device)
    print(f"Using threshold: {threshold}")

    loader = build_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    model = load_model(args.model_path, device)
    y_prob, y_true, pids = infer(model, loader, device)

    save_predictions(y_prob, y_true, pids, threshold, args.output_dir)

    metrics = calc_metrics(y_true, y_prob, threshold)
    print("\n--- Evaluation Results ---")
    for k, v in metrics.items():
        print(f"{k:10}: {v:.4f}")
