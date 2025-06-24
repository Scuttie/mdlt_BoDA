import argparse
import os
import pandas as pd


def imbalance_ratio(counts: pd.Series) -> float:
    return counts.max() / counts.min()


def compute_ratios(csv_path: str):
    df = pd.read_csv(csv_path)
    df_train = df[df['split'] == 'train']

    label_counts = df_train['label'].value_counts()
    dataset_ratio = imbalance_ratio(label_counts)

    env_ratios = {}
    for env in sorted(df_train['env'].unique()):
        env_counts = df_train[df_train['env'] == env]['label'].value_counts()
        env_ratios[env] = imbalance_ratio(env_counts)
    return dataset_ratio, env_ratios


def main():
    parser = argparse.ArgumentParser(description='Compute imbalance ratios for MDLT datasets.')
    parser.add_argument('--split_dir', default='mdlt/dataset/split', help='Directory containing CSV splits')
    args = parser.parse_args()

    datasets = ['DomainNet', 'OfficeHome', 'PACS', 'TerraIncognita', 'VLCS']
    for dset in datasets:
        csv_path = os.path.join(args.split_dir, f'{dset}.csv')
        ratio, env_ratios = compute_ratios(csv_path)
        print(f'{dset}: {ratio:.2f}')
        for env, r in env_ratios.items():
            print(f'  {env}: {r:.2f}')
        print()


if __name__ == '__main__':
    main()
