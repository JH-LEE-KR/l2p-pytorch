#!/bin/bash

#SBATCH --job-name=l2p
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p batch
#SBATCH -w agi1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=20G
#SBATCH --time=8:00:00
#SBATCH -o %x_%j.out
#SBTACH -e %x_%j.err

source /data/jaeho/init.sh
conda activate torch38gpu
python main.py