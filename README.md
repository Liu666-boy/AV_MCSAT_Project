基于跨模态注意力融合的声振联合轴承故障诊断网络

本模型在同工况数据上复现声学基线、振动基线和声振融合模型。

一 文件内容
- `train.py`：训练并测试入口。
- `models/`：原始 MCSAT 单输入骨干和 MCSAT_AV 双流跨注意力融合模型。
- `datasets/cond1_av.py`：按录音文件配对 acoustic__va (N).csv 与 va (N).csv，并按类别做 80/20 文件级划分。
- `helper/`：训练、测试、优化器和统计工具。
- `Cond_1_vibration_4/3rpm/`：Health、IRC、IRS、IRW、ORC、ORS、ORW 七类，共 336 个 CSV（168 对声学/振动文件）。
- `run_tests.ps1`、`run_tests.sh`：三套可直接运行的对比命令。
- `predict_av.py`：加载声学文件和对应振动文件做单对录音预测。

二 环境与部署

1.下载解压后，打开项目文件，在所用的服务器安装 requirements-test.txt 中的依赖，并按服务器 CUDA 版本安装 PyTorch。

2.将项目文件放到所用服务器内。

3.复制上面给出的运行脚本，然后粘贴到服务器终端按回车即可训练与测试，运行完毕后会得到验收记录。

4.复制上面给出的权重脚本，然后粘贴到服务器终端按回车即可测试

三 训练与测试运行脚本

声学基线

```bash
python train.py --datasets cond1_same_condition --work_dir ./Cond_1_vibration_4 --condition 3rpm --modality acoustic --model MCSAT -ic 1 --num_cls 7 --length 1024 --step 1024 --batch_size 32 --epochs 30 --lr 0.00003 --lr_scheduler cos --train_batches_per_epoch 100 --test_batches 55 --test_fraction 0.2 --trail 1
```

 振动基线

```bash
python train.py --datasets cond1_same_condition --work_dir ./Cond_1_vibration_4 --condition 3rpm --modality vibration --model MCSAT -ic 1 --num_cls 7 --length 1024 --step 1024 --batch_size 32 --epochs 30 --lr 0.00003 --lr_scheduler cos --train_batches_per_epoch 100 --test_batches 55 --test_fraction 0.2 --trail 1
```

声振融合

```bash
python train.py --datasets cond1_same_condition --work_dir ./Cond_1_vibration_4 --condition 3rpm --modality av --model MCSAT_AV -ic 1 --num_cls 7 --length 1024 --step 1024 --batch_size 32 --epochs 30 --lr 0.00003 --lr_scheduler cos --train_batches_per_epoch 100 --test_batches 55 --test_fraction 0.2 --trail 1
```

四 权重运行脚本

声学基线

```bash
python train.py --eval_only --checkpoint weights/MCSAT_Cond1_3rpm_acoustic_epoch30.pth --datasets cond1_same_condition --work_dir ./Cond_1_vibration_4 --condition 3rpm --modality acoustic --model MCSAT -ic 1 --num_cls 7 --length 1024 --step 1024 --batch_size 32 --test_fraction 0.2 --test_batches 55 --trail 1
```

 振动基线

```bash
python train.py --eval_only --checkpoint weights/MCSAT_Cond1_3rpm_vibration_epoch30.pth --datasets cond1_same_condition --work_dir ./Cond_1_vibration_4 --condition 3rpm --modality vibration --model MCSAT -ic 1 --num_cls 7 --length 1024 --step 1024 --batch_size 32 --test_fraction 0.2 --test_batches 55 --trail 1
```

声振融合

```bash
python train.py --eval_only --checkpoint weights/MCSAT_AV_Cond1_vibration_4_3rpm_epoch30.pt --datasets cond1_same_condition --work_dir ./Cond_1_vibration_4 --condition 3rpm --modality av --model MCSAT_AV -ic 1 --num_cls 7 --length 1024 --step 1024 --batch_size 32 --test_fraction 0.2 --test_batches 55 --train_batches_per_epoch 100 --trail 1
```

五 对比结果

| 模型 | 输入 | L2 准确率 | FPR | FNR | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| MCSAT | 声学 | 96.9886% | 3.0682%（27/880） | 0.9091%（8/880） | 0.9705 | 0.9699 | 0.9699 |
| MCSAT | 振动 | 94.3182% | 0.7955%（7/880） | 0.7955%（7/880） | 0.9437 | 0.9432 | 0.9428 |
| MCSAT_AV | 声振融合 | 99.8864% | 0.0000%（0/880） | 0.0000%（0/880） | 0.9989 | 0.9989 | 0.9989 |

六 数据集
本文用到的数据集是风力涡轮机在时变负荷下螺距轴承的振动和声学数据（Cond_1，Cond_2）
数据集链接：https://data.mendeley.com/datasets/2n9jkd384k/1



