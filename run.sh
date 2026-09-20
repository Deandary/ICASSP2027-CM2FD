#!/bin/bash

#SBATCH --partition=gpu-l20
### 指定队列为gpu

#SBATCH --nodes=1
#SBATCH --nodelist=gpu-l20-1
### 指定该作业需要1个节点数

#SBATCH --ntasks-per-node=8
### 每个节点所运行的进程数为52

### SBATCH --ntasks=16
### 该作业需要16个CPU




#SBATCH --gres=gpu:1 
###（声明需要的GPU数量）【单节点最大申请2个GPU】
source /opt/app/anaconda3/bin/activate
conda activate SDCL

cd /home/qsdeng/workdir/projects/gpu_l20_2

python3 main.py --dataset sysu --debug wsl  --save-path sysu_agw_324_0.2_1 --arch cssp --stage1-epoch 20 --milestone 30 70 --lr 0.0003 --device 0