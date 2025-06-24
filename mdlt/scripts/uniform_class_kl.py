import argparse
import os
import pandas as pd
import numpy as np


def kl_divergence(counts: pd.Series, label_set=None) -> float:
    if label_set is None:
        label_set = counts.index
    probs = counts.reindex(label_set, fill_value=0).astype(float).values
    probs = probs / probs.sum()
    uniform = np.ones(len(label_set)) / len(label_set)
    eps = 1e-12
    return float((probs * np.log((probs + eps) / (uniform + eps))).sum())


def compute_kl(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df[df['split'] == 'train']
    label_set = sorted(df['label'].unique())

    dataset_counts = df['label'].value_counts().sort_index()
    dataset_kl = kl_divergence(dataset_counts, label_set)

    env_kls = {}
    for env, group in df.groupby('env'):
        counts = group['label'].value_counts().sort_index()
        env_kls[env] = kl_divergence(counts, label_set)
    return dataset_kl, env_kls


def main():
    parser = argparse.ArgumentParser(
        description='KL divergence of class distributions vs uniform')
    parser.add_argument('--split_dir', default='mdlt/dataset/split',
                        help='Directory containing CSV splits')
    args = parser.parse_args()

    datasets = ['DomainNet', 'OfficeHome', 'PACS', 'TerraIncognita', 'VLCS']
    for dset in datasets:
        csv_path = os.path.join(args.split_dir, f'{dset}.csv')
        kl_dataset, env_kls = compute_kl(csv_path)
        print(f'{dset}: {kl_dataset:.4f}')
        for env, kl in env_kls.items():
            print(f'  {env}: {kl:.4f}')
        print()


if __name__ == '__main__':
    main()
