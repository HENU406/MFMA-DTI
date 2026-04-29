from utils import *
from model import *
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl.nn.pytorch as dglnn
from sklearn.metrics import roc_auc_score, f1_score
import warnings
import os
import argparse

warnings.filterwarnings("ignore")
set_random_seed(52)

args = argparse.ArgumentParser()
args.add_argument("--learning_rate", type=float, default=0.001)
args.add_argument("--inp_size", type=int, default=128)  # 随机特征的维度
args.add_argument("--hidden_size", type=int, default=128)
args.add_argument("--out_size", type=int, default=128)
args.add_argument("--dropout", type=float, default=0.2)
args.add_argument("--ptwd", type=float, default=1e-3)
args.add_argument("--epochs", type=int, default=200)
args.add_argument("--dataname", type=str, default="heter")
args.add_argument("--kfold", type=int, default=5, help="number of folds")
args.add_argument("--alpha_g", type=float, default=0.4, help="Graph-level mask rate")
args.add_argument("--alpha_n", type=float, default=0.3, help="Neighbor-level mask rate")


args = args.parse_args()
args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

def train_kfold(epochs, k=5):
    print("=" * 70)
    print(f"Dataset: {args.dataname}")
    print("=" * 70)

    data, graph, num, all_meta_paths= load_dataset(args.dataname)
    graph = [i.to(args.device) for i in graph]
    label = torch.tensor(data[:, 2:3]).to(args.device)

    rand_d = torch.randn((num[0], args.inp_size)).to(args.device)
    rand_p = torch.randn((num[1], args.inp_size)).to(args.device)
    random_feature = [rand_d, rand_p]

    try:
        drug_npz = f"data/{args.dataname}/drug_features.npz"
        prot_npz = f"data/{args.dataname}/protein_features.npz"

        d_data = np.load(drug_npz)
        p_data = np.load(prot_npz)

        feat_d = torch.from_numpy(d_data['features']).float().to(args.device)
        feat_p = torch.from_numpy(p_data['features']).float().to(args.device)

        d_dim = feat_d.shape[1]
        p_dim = feat_p.shape[1]

    except Exception as e:
        feat_d = torch.randn((num[0], args.inp_size)).to(args.device)
        feat_p = torch.randn((num[1], args.inp_size)).to(args.device)
        d_dim = args.inp_size
        p_dim = args.inp_size

    pretrained_feature = [feat_d, feat_p]

    folds = get_kfold_data(data, k=k, seed=52)
    fold_results = {'acc': [], 'roc': [], 'pr': []}

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n========== Fold {fold_idx + 1} / {k} ==========")

        model = MFMA_DTI(
            all_meta_paths=all_meta_paths,
            in_size=args.inp_size,
            hidden_size=args.hidden_size,
            out_size=args.out_size,
            dropout=args.dropout,
            layersnums=1,
            batch_size_limit=512,
            feature_dims=[d_dim, p_dim],
            alpha_g=args.alpha_g,
            alpha_n=args.alpha_n,
        ).to(args.device)

        optim = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.ptwd
        )
        ce_loss_fn = torch.nn.CrossEntropyLoss()

        best_acc = 0
        best_roc = 0
        best_pr = 0
        curr_acc = curr_roc = 0

        pbar = tqdm.tqdm(range(epochs), desc=f"Fold {fold_idx + 1}")
        for e in pbar:
            model.train()
            train_data = data[train_idx]

            out, contrast_loss = model(
                graph, random_feature, pretrained_feature,
                train_data, torch.arange(len(train_idx)).to(args.device),
                iftrain=True
            )

            total_loss = ce_loss_fn(out, label[train_idx].reshape(-1)) + contrast_loss

            with torch.no_grad():
                train_acc = (
                    (out.argmax(dim=1) == label[train_idx].reshape(-1))
                    .float().mean().item()
                )

            optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            model.eval()
            with torch.no_grad():
                out_test, _ = model(
                    graph, random_feature, pretrained_feature,
                    data, test_idx, iftrain=False, e=e
                )
                out_test = out_test[test_idx]

                curr_acc = (
                    (out_test.argmax(dim=1) == label[test_idx].reshape(-1))
                    .float().mean().item()
                )
                curr_roc = get_roc(out_test, label[test_idx])
                curr_pr = get_pr(out_test, label[test_idx])

                if curr_roc > best_roc:
                    best_acc, best_roc, best_pr = curr_acc, curr_roc, curr_pr
                    os.makedirs("checkpoints", exist_ok=True)
                    torch.save(
                        model.state_dict(),
                        f"checkpoints/{args.dataname}_fold{fold_idx + 1}.pt"
                    )

                pbar.set_postfix(best_roc=f"{best_roc:.4f}", best_pr=f"{best_pr:.4f}")

        fold_results['acc'].append(best_acc)
        fold_results['roc'].append(best_roc)
        fold_results['pr'].append(best_pr)
        pbar.close()  # 显式关闭本折的进度条
        print(f"Fold {fold_idx + 1} Finished. Best ROC: {best_roc:.4f}")

    # ── 5. 汇总结果 ──────────────────────────────────────────────────────
    avg_acc = np.mean(fold_results['acc']);std_acc = np.std(fold_results['acc'])
    avg_roc = np.mean(fold_results['roc']);std_roc = np.std(fold_results['roc'])
    avg_pr = np.mean(fold_results['pr']);std_pr = np.std(fold_results['pr'])

    print("\n" + "=" * 70)
    print(f"Final {k}-Fold Results Summary")
    print(f"ACC : {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"ROC : {avg_roc:.4f} ± {std_roc:.4f}")
    print(f"PR  : {avg_pr:.4f}  ± {std_pr:.4f}")
    print("=" * 70)

    print("\n每折详细结果:")
    print(f"{'Fold':<6} {'ACC':<10} {'ROC':<10} {'PR':<10}")
    print("-" * 40)
    for i in range(k):
        print(f"{i + 1:<6} {fold_results['acc'][i]:<10.4f} "
              f"{fold_results['roc'][i]:<10.4f} {fold_results['pr'][i]:<10.4f}")
    print("-" * 40)
    print(f"{'Mean':<6} {avg_acc:<10.4f} {avg_roc:<10.4f} {avg_pr:<10.4f}")
    print(f"{'Std':<6} {std_acc:<10.4f} {std_roc:<10.4f} {std_pr:<10.4f}")

if __name__ == "__main__":
    train_kfold(args.epochs, k=args.kfold)