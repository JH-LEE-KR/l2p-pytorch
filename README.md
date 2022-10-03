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
To train a model on CIFAR-100, set the `--data-path` (path to dataset) and `--output-dir` (result logging directory) and other options in `train.sh` properly and run in <a href="https://slurm.schedmd.com/documentation.html">Slurm</a> system.

## Training
To train a model on CIFAR-100 via command line:

Single node with single gpu
```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model vit_base_patch16_224 --batch-size 16 --data-path /local_datasets/ --output_dir ./output --epochs 5
```

Single node with multi gpus
```
python -m torch.distributed.launch --nproc_per_node=<Num GPUs> --use_env main.py --model vit_base_patch16_224 --batch-size 16 --data-path /local_datasets/ --output_dir ./output --epochs 5
```

Also available in Slurm by changing options on `train.sh`

### Multinode train

Distributed training is available via Slurm and [submitit](https://github.com/facebookincubator/submitit):

```
pip install submitit
```

To train a model on CIFAR-100 on 2 nodes with 4 gpus each:

```
python run_with_submitit.py --shared_folder <Absolute Path of shared folder for all nodes>
```
Absolute Path of shared folder must be accessible from all nodes.<br>
According to your environment, you can use `NCLL_SOCKET_IFNAME=<Your own IP interface to use for communication>` optionally.

## Evaluation
To evaluate a trained model:
```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --eval
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

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

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
