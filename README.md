# LCASAFormer
Cross-Attention Enhanced Backbone Network for 3D Point Cloud Tasks

## Install

### Requirements

- `Ubuntu 20.04`
- `Anaconda` with `python=3.6`
- `pytorch==1.7.1`
- `torchvision` with  `pillow<7`
- `cuda=11.4`
- `trimesh>=2.35.39,<2.35.40`
- `'networkx>=2.2,<2.3'`
- compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used in the backbone
  network: `sh init.sh`
- others: `pip install termcolor opencv-python tensorboard`

### Data preparation

For SUN RGB-D, follow the [README](https://github.com/zeliu98/Group-Free-3D/blob/master/sunrgbd/README.md) under the `sunrgbd` folder.

## Usage

#### SUN RGB-D

For  training:

```bash
python -m torch.distributed.launch --master_port <port_num> --nproc_per_node <num_of_gpus_to_use> \
    train_dist.py --max_epoch 600 --lr_decay_epochs 420 480 540 --num_point 20000 --num_decoder_layers 6 \
    --size_cls_agnostic --size_delta 0.0625 --heading_delta 0.04 --center_delta 0.1111111111111 \
    --learning_rate 0.004 --decoder_learning_rate 0.0002 --weight_decay 0.00000001 --query_points_generator_loss_coef 0.2 --obj_loss_coef 0.4 \
    --dataset sunrgbd --data_root <data directory> [--log_dir <log directory>]
```

For  evaluation:

```bash
python eval_avg.py --num_point 20000 --num_decoder_layers 6 --size_cls_agnostic \
    --checkpoint_path <checkpoint> --avg_times 5 \
    --dataset sunrgbd --data_root <data directory> [--dump_dir <dump directory>]
```

