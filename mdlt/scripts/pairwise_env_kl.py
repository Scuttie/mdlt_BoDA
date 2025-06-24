import argparse
import os
import pandas as pd
import numpy as np
import itertools


def kl_divergence(p_counts: pd.Series, q_counts: pd.Series) -> float:
    labels = sorted(set(p_counts.index).union(q_counts.index))
    p = p_counts.reindex(labels, fill_value=0).astype(float).values
    q = q_counts.reindex(labels, fill_value=0).astype(float).values
    p = p / p.sum()
    q = q / q.sum()
    eps = 1e-12
    return float((p * np.log((p + eps) / (q + eps))).sum())


def compute_pairwise(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df[df['split'] == 'train']

    counts = {
        env: group['label'].value_counts().sort_index()
        for env, group in df.groupby('env')
    }
    envs = sorted(counts)

    pairs = []
    for a, b in itertools.combinations(envs, 2):
        kl_ab = kl_divergence(counts[a], counts[b])
        kl_ba = kl_divergence(counts[b], counts[a])
        sym_kl = 0.5 * (kl_ab + kl_ba)
        pairs.append((a, b, sym_kl))
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Pairwise KL divergence between environments')
    parser.add_argument('--split_dir', default='mdlt/dataset/split', help='Directory containing CSV splits')
    args = parser.parse_args()

    datasets = ['DomainNet', 'OfficeHome', 'PACS', 'TerraIncognita', 'VLCS']
    for dset in datasets:
        csv_path = os.path.join(args.split_dir, f'{dset}.csv')
        print(dset)
        for env_a, env_b, kl in compute_pairwise(csv_path):
            print(f'  {env_a} vs {env_b}: {kl:.4f}')
        print()


if __name__ == '__main__':
    main()
