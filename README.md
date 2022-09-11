# L2P Pytorch Implementation

This repository contains PyTorch implementation code for awesome continual learning method <a href="https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Learning_To_Prompt_for_Continual_Learning_CVPR_2022_paper.pdf">L2P</a>, proposed in Wang, Zifeng, et al. "Learning to prompt for continual learning." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

The official Jax implementation is <a href="https://github.com/google-research/l2p">here</a>.

## Environment
The system I used and tested in
- Ubuntu 20.04.4 LTS
- Slurm 21.08.1
- NVIDIA GeForce RTX 3090
- Python 3.8

## Usage
First, clone the repository locally:
```
git clone https://github.com/Lee-JH-KR/l2p-pytorch
cd l2p-pytorch
```
Then, install the packages below:
```
pytorch==1.12.1
torchvision==0.13.1
timm==0.6.7
pillow==9.2.0
matplotlib==3.5.3
torchprofile==0.0.4
```
These packages can be installed easily by 
```
pip install -r requirements.txt
```

## Data preparation
If you already have CIFAR-100 datasets, pass your dataset path to  `--data-path`.


If the dataset isn't ready, change the download argument in `continual_dataloader.py` as follows
```
datasets.CIFAR100(download=True)
```

## Train
To train a model on CIFAR-100, set the `--data-path` (path to dataset) and `--output-dir` (result logging directory) in train.sh properly and run in <a href="https://slurm.schedmd.com/documentation.html">Slurm</a> system or `bash ./train.sh`.

## Evaluation
To evaluate a trained model:
```
python main.py --eval 
```
## Result
Test results on a single GPU.
| Name | Acc@1 | Forgetting |
| --- | --- | --- |
| Pytorch-Implementation | 82.77 | 6.43 |
| Reproduce Official-Implementation | 82.59 | 7.88 |

Here are the metrics used in the test, and their corresponding meanings:

| Metric | Description |
| ----------- | ----------- |
| Acc@1  | Average evaluation accuracy up until the last task |
| Forgetting | Average forgetting up until the last task |


## Throughput
You can measure the throughput of the model by passing `--speed_test` and optionally `--speed_test_only` to `main.py`.

## Cite
```
@inproceedings{wang2022learning,
  title={Learning to prompt for continual learning},
  author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={139--149},
  year={2022}
}
```
