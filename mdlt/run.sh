#!/bin/bash

#SBATCH --job-name=BODA5_SWEEP_RES18
#SBATCH --partition=laal_a6000
#SBATCH --nodes=1    
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=16
#SBATCH --output=./slurm_logs/S-%x.%j.out     

cd /home/hyunggyu/imbalance/multi-domain-imbalance

# nvidia-smi

# python -m mdlt.train \
#   --dataset PACS \
#   --algorithm IRM \
#   --output_folder_name res18_test \
#   --data_dir /home/shared \
#   --output_dir ./output \
#   --hparams '{"resnet18": true}'


# python -m mdlt.scripts.download --data_dir /home/shared

# sweep 실행
# python -m mdlt.sweep launch \
#   --output_folder_name sweep_res18_mydataset2 \
#   --data_dir /home/shared \
#   --output_dir ./output \
#   --hparams '{"resnet18": true}' \
#   --skip_confirmation

# ALGORITHMS list
# --algorithms 'ERM' \
# --algorithms 'IRM' \
# --algorithms 'GroupDRO' \
# --algorithms 'Mixup' \
# --algorithms 'MLDG' \
# --algorithms 'CORAL' \
# --algorithms 'MMD' \
# --algorithms 'DANN' \
# --algorithms 'CDANN' \
# --algorithms 'MTL' \
# --algorithms 'SagNet' \
# --algorithms 'Fish' \
# --algorithms 'ReSamp' \
# --algorithms 'ReWeight' \
# --algorithms 'SqrtReWeight' \
# --algorithms 'CBLoss' \
# --algorithms 'Focal' \
# --algorithms 'LDAM' \
# --algorithms 'BSoftmax' \
# --algorithms 'CRT' \
# --algorithms 'BoDA' \

# INCOMPLETE 디렉토리 삭제하고 재실행
# Step 1: INCOMPLETE된 실험 디렉토리 삭제
python -m mdlt.sweep delete_incomplete \
  --output_folder_name sweep_res18_mydataset2 \
  --algorithms 'BoDA' \
  --data_dir /home/shared \
  --output_dir ./output \
  --hparams '{"resnet18": true}' \
  --skip_confirmation

# ALG별 SWEEP 실행
python -m mdlt.sweep launch \
  --output_folder_name sweep_res18_mydataset2 \
  --algorithms 'BoDA' \
  --data_dir /home/shared \
  --output_dir ./output \
  --hparams '{"resnet18": true}' \
  --skip_confirmation

# collect 실행
# python -m mdlt.scripts.collect_results \
#   --input_dir /home/hyunggyu/imbalance/multi-domain-imbalance/output/sweep_res18_mydataset2
  


# INCOMPLETE 디렉토리 삭제하고 재실행
# Step 1: INCOMPLETE된 실험 디렉토리 삭제
# python -m mdlt.sweep delete_incomplete \
#   --output_folder_name sweep_res18_mydataset2 \
#   --data_dir /home/shared \
#   --output_dir ./output \
#   --hparams '{"resnet18": true}' \
#   --skip_confirmation

  # --algorithms 'BODA' \
# Step 2: 다시 launch



