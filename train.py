import os
import torch
import argparse
import time
import pickle
import random
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

# 保留原作者的核心组件导入
from helper.loops import train_one_epoch, validate
from models.backbone import model_dict  # 🌟 因为你删了 Signal 文件夹
from helper.create import create_optimizer, create_scheduler # 🌟 去掉 creat_loss
from torch.nn import CrossEntropyLoss # 🌟 直接引入官方损失函数
from thop import profile

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import itertools
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser('the argument for training')

    # regular parameters
    parser.add_argument("--print_freq", type=int, default=25, help="the frequency to print")
    parser.add_argument("--save_freq", type=int, default=100, help="the frequency to save")
    parser.add_argument('--batch_size', type=int, default=32, help="the batch size")
    parser.add_argument("--epochs", type=int, default=100, help="the total train epoch")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="trained checkpoint to load for evaluation")
    parser.add_argument("--eval_only", action="store_true",
                        help="skip training and evaluate --checkpoint on the test split")

    # optimizer parameters
    parser.add_argument("--optimizer_name", type=str, default="adamw", choices=["adam", "sgd", "adamw", "rmsprop"])
    parser.add_argument("--opt_eps", type=float, default=1e-8, help="Optimizer Epsilon")
    parser.add_argument("--opt_betas", type=str, default=None, help="Optimizer Betas, use opt default")
    parser.add_argument("--momentum", type=float, default=0.9, help="Optimizer momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="the norm to weight")
    parser.add_argument("--lr", type=float, default=8 * (1e-5), help="the init learning rate")
    parser.add_argument("--amp", type=int, default=0, choices=[0, 1], help="using amp to train or not")

    # schedule parameters
    parser.add_argument("--lr_scheduler", type=str, default="cos", choices=["step", "mstep", "exp", "cos", "reduce"])
    parser.add_argument("--lr_decay_epochs", type=str, default="50,80,120")
    parser.add_argument("--lr_decay_rate", type=float, default=0.95)

    # loss parameters
    parser.add_argument("--loss_name", type=str, default="cross_entropy",
                        choices=["cross_entropy", "smooth_cross_entropy", "jsd_loss", "enhanced_loss"])

    # ====== 🌟 新增：加入 hust_custom 和 jnu_custom ======
    parser.add_argument("--work_dir", type=str, required=True, help="数据集根目录路径")
    parser.add_argument("--datasets", type=str, default="cwru_custom", choices=["cwru_custom", "wt_custom", "hust_custom", "jnu_custom", "uods_vafdc", "cond1_acoustic", "cond1_paired", "cond1_av", "cond1_av_same_condition", "cond1_same_condition", "fstf_stethoscope"])
    parser.add_argument("--train_conditions", type=str, default="0hp,1hp,2hp", help="训练工况，用逗号分隔")
    parser.add_argument("--test_condition", type=str, default="3hp", help="测试（目标）工况")

    # CWRU 默认10分类, WT齿轮箱 默认5分类, HUST 默认7分类, JNU 默认4分类
    parser.add_argument("--num_cls", type=int, default=10, help="分类类别数")
    parser.add_argument("--length", type=int, default=2048, help="样本长度")
    parser.add_argument("--step", type=int, default=1024, help="滑动窗口步长")
    parser.add_argument("--test_fraction", type=float, default=0.2, help="UODS 每类按文件留出的测试比例")
    parser.add_argument("--condition", type=str, default="1rpm", choices=["1rpm", "3rpm"], help="同工况训练/测试时使用的单一工况")
    parser.add_argument("--modality", type=str, default="av", choices=["acoustic", "vibration", "av"], help="同工况 Cond_1 输入：声学单模态、振动单模态或声振融合")
    parser.add_argument("--channel", type=int, default=0, help="UODS MAT 矩阵通道索引")
    parser.add_argument("--acoustic_channel", type=int, default=0, help="Cond_1 传感器列索引，0-4 对应 CSV 第2-6列")
    parser.add_argument("--signal_type", type=str, default="acoustic", choices=["acoustic", "vibration"], help="配对 Cond_1 数据使用声学或单路振动信号")
    parser.add_argument("--train_batches_per_epoch", type=int, default=100, help="Cond_1 每轮最多训练批次数")
    parser.add_argument("--test_batches", type=int, default=55, help="Cond_1 测试时最多评估批次数")
    parser.add_argument("-t", "--trail", type=int, default=1, help="the experiment id")

    parser.add_argument("--model", type=str, default="MCSAT", help="the name of model")
    parser.add_argument("-ic", "--input_channel", type=int, default=1, help="the input channel of input data")
    parser.add_argument("--layer_args", type=str, default='100,64,32', help="the hidden layer neurons")

    opt = parser.parse_args()

    # MultiStepLR and the legacy adjust_lr helper both expect integer epochs.
    opt.lr_decay_epochs = [int(epoch.strip()) for epoch in opt.lr_decay_epochs.split(",") if epoch.strip()]

    if opt.datasets == "uods_vafdc":
        opt.num_cls = 5
    elif opt.datasets == "cond1_acoustic":
        opt.num_cls = 7
    elif opt.datasets == "cond1_paired":
        opt.num_cls = 7
    elif opt.datasets == "cond1_av":
        opt.num_cls = 7
    elif opt.datasets == "cond1_av_same_condition":
        opt.num_cls = 7
    elif opt.datasets == "cond1_same_condition":
        opt.num_cls = 7
    elif opt.datasets == "fstf_stethoscope":
        opt.num_cls = 6

    if not opt.layer_args:
        opt.h_args = None
    else:
        opt.h_args = [int(it) for it in opt.layer_args.split(",")]

    if opt.opt_betas:
        opt.betas = [float(it) for it in opt.opt_betas.split(",")]
    else:
        opt.betas = None

    opt.train_size = None

    if opt.datasets in {"cond1_av_same_condition", "cond1_same_condition"}:
        modality_name = f"_{opt.modality}" if opt.datasets == "cond1_same_condition" else ""
        opt.model_name = (f'{opt.model}_{opt.datasets}{modality_name}_Condition{opt.condition}'
                          f'_test{opt.test_fraction}_lr_{opt.lr}_trial_{opt.trail}')
    else:
        opt.model_name = f'{opt.model}_{opt.datasets}_TestOn{opt.test_condition}_lr_{opt.lr}_trial_{opt.trail}'

    opt.save_path = './save/model'
    opt.save_folder = os.path.join(opt.save_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.tb_path = './save/tensorboard'
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    return opt


def main():
    opt = parse_args()

    # Reuse the same run seed for comparable modality splits and repeatability.
    random.seed(opt.trail)
    np.random.seed(opt.trail)
    torch.manual_seed(opt.trail)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.trail)

    train_result = {'accuracy': [], 'loss': [], 'lr': []}
    test_result = {'accuracy': [], 'loss': []}
    val_result = {'accuracy': [], 'loss': []}
    best_acc, best_epoch = float("-inf"), 0
    total_train_time, total_test_time, total_val_time = 0, 0, 0

    if opt.datasets in {"cond1_av", "cond1_av_same_condition"} and opt.model != "MCSAT_AV":
        raise ValueError(f"datasets={opt.datasets} requires --model MCSAT_AV")
    if opt.datasets == "cond1_same_condition":
        expected_model = "MCSAT_AV" if opt.modality == "av" else "MCSAT"
        if opt.model != expected_model:
            raise ValueError(f"--modality {opt.modality} requires --model {expected_model}")
    model = model_dict[opt.model](h_args=opt.h_args, in_c=opt.input_channel, num_cls=opt.num_cls)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("Get flops and params==> ......")
    input_data = torch.randn(1, opt.input_channel, opt.length, device=device)
    profile_inputs = (input_data, input_data) if opt.model == "MCSAT_AV" else (input_data,)
    flops, params = profile(model, inputs=profile_inputs)
    print(f'flops: {flops} | params: {params}')

    if opt.datasets == "cond1_same_condition":
        print(f"Loading same-condition data (condition={opt.condition}, modality={opt.modality}; "
              f"file-level train/test={1.0 - opt.test_fraction:.0%}/"
              f"{opt.test_fraction:.0%}) ==> ......")
    elif opt.datasets == "cond1_av_same_condition":
        print(f"Loading data (same condition: {opt.condition}; file-level split: "
              f"train/test={1.0 - opt.test_fraction:.0%}/{opt.test_fraction:.0%}) ==> ......")
    else:
        print(f"Loading data (Train on: {opt.train_conditions} | Test on: {opt.test_condition}) ==> ......")
    train_conds = opt.train_conditions.split(',')
    val_loader = None

    if opt.datasets == "cond1_av":
        from datasets.cond1_av import get_cond1_av_loaders
        train_loader, test_loader = get_cond1_av_loaders(
            root=opt.work_dir, train_conditions=opt.train_conditions,
            test_condition=opt.test_condition, batch_size=opt.batch_size,
            length=opt.length, step=opt.step,
            train_batches_per_epoch=opt.train_batches_per_epoch,
            test_batches=opt.test_batches, seed=opt.trail)
    elif opt.datasets == "cond1_av_same_condition":
        from datasets.cond1_av import get_cond1_av_same_condition_loaders
        train_loader, test_loader = get_cond1_av_same_condition_loaders(
            root=opt.work_dir, condition=opt.condition, batch_size=opt.batch_size,
            length=opt.length, step=opt.step, test_fraction=opt.test_fraction,
            train_batches_per_epoch=opt.train_batches_per_epoch,
            test_batches=opt.test_batches, seed=opt.trail)
    elif opt.datasets == "cond1_same_condition":
        from datasets.cond1_av import get_cond1_same_condition_loaders
        train_loader, test_loader = get_cond1_same_condition_loaders(
            root=opt.work_dir, condition=opt.condition, modality=opt.modality,
            batch_size=opt.batch_size, length=opt.length, step=opt.step,
            test_fraction=opt.test_fraction,
            train_batches_per_epoch=opt.train_batches_per_epoch,
            test_batches=opt.test_batches, seed=opt.trail)
    elif opt.datasets == "cwru_custom":
        from datasets.cwru_custom import get_cross_condition_dataloader
        train_loader, test_loader = get_cross_condition_dataloader(
            data_dir=opt.work_dir,
            train_conditions=train_conds,
            test_condition=opt.test_condition,
            batch_size=opt.batch_size,
            sample_length=opt.length,
            stride=opt.step
        )
    elif opt.datasets == "wt_custom":
        from datasets.wt_custom import get_cross_condition_gearbox_dataloader
        train_loader, test_loader, _ = get_cross_condition_gearbox_dataloader(
            data_dir=opt.work_dir,
            train_conditions=train_conds,
            test_condition=opt.test_condition,
            batch_size=opt.batch_size,
            sample_length=opt.length,
            stride=opt.step
        )
    elif opt.datasets == "hust_custom":
        from datasets.hust_custom import get_cross_condition_hust_dataloader
        train_loader, test_loader, _ = get_cross_condition_hust_dataloader(
            data_dir=opt.work_dir,
            train_conditions=train_conds,
            test_condition=opt.test_condition,
            batch_size=opt.batch_size,
            sample_length=opt.length,
            stride=opt.step
        )
    # ====== 🌟 新增：JNU 数据集的分支 ======
    elif opt.datasets == "jnu_custom":
        from datasets.jnu_custom import get_cross_condition_jnu_dataloader
        train_loader, test_loader, _ = get_cross_condition_jnu_dataloader(
            data_dir=opt.work_dir,
            train_conditions=train_conds,
            test_condition=opt.test_condition,
            batch_size=opt.batch_size,
            sample_length=opt.length,
            stride=opt.step
        )
    elif opt.datasets == "uods_vafdc":
        from datasets.uods_vafdc import get_uods_vafdc_dataloader
        train_loader, test_loader, _ = get_uods_vafdc_dataloader(
            data_dir=opt.work_dir,
            batch_size=opt.batch_size,
            sample_length=opt.length,
            stride=opt.step,
            test_fraction=opt.test_fraction,
            channel=opt.channel,
            seed=opt.trail
        )
    elif opt.datasets == "cond1_acoustic":
        from datasets.cond1_acoustic import get_cond1_acoustic_dataloader
        train_loader, test_loader, _ = get_cond1_acoustic_dataloader(
            data_dir=opt.work_dir,
            train_conditions=train_conds,
            test_condition=opt.test_condition,
            batch_size=opt.batch_size,
            sample_length=opt.length,
            stride=opt.step,
            channel=opt.acoustic_channel,
            seed=opt.trail,
            train_batches_per_epoch=opt.train_batches_per_epoch,
            test_batches=opt.test_batches
        )
    elif opt.datasets == "cond1_paired":
        from datasets.cond1_paired import get_cond1_paired_dataloader
        train_loader, test_loader, _ = get_cond1_paired_dataloader(
            data_dir=opt.work_dir,
            train_conditions=train_conds,
            test_condition=opt.test_condition,
            signal_type=opt.signal_type,
            batch_size=opt.batch_size,
            sample_length=opt.length,
            stride=opt.step,
            seed=opt.trail,
            train_batches_per_epoch=opt.train_batches_per_epoch,
            test_batches=opt.test_batches
        )
    elif opt.datasets == "fstf_stethoscope":
        from datasets.fstf_stethoscope import get_fstf_stethoscope_dataloader
        train_loader, test_loader, _ = get_fstf_stethoscope_dataloader(
            data_dir=opt.work_dir,
            train_conditions=train_conds,
            test_condition=opt.test_condition,
            batch_size=opt.batch_size,
            sample_length=opt.length,
            stride=opt.step
        )
    else:
        raise ValueError("Unsupported dataset type!")

    # Evaluation-only path: keep the normal training path unchanged, but allow
    # a project user to reuse a delivered weight without retraining.
    if opt.eval_only:
        if not opt.checkpoint:
            raise ValueError("--eval_only requires --checkpoint <path>")
        checkpoint = torch.load(opt.checkpoint, map_location=device)
        if isinstance(checkpoint, dict):
            for key in ("model", "state_dict", "model_state_dict"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported checkpoint format: {opt.checkpoint}")
        # Checkpoints produced by DataParallel commonly carry a module. prefix.
        state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in checkpoint.items()
        }
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded checkpoint: {opt.checkpoint}")
        print("Evaluation-only mode: training is skipped.")
        evaluate_and_plot_confusion_matrix(model, test_loader, opt)
        return

    optimizer = create_optimizer(model, opt)
    lr_scheduler = create_scheduler(optimizer, opt)
    criterion = CrossEntropyLoss().to(device)  # 🌟 直接使用官方标准交叉熵
    scaler = GradScaler()
    tb_writter = SummaryWriter(log_dir=opt.tb_folder)

    init_data = torch.zeros((1, opt.input_channel, opt.length), device=device)
    tb_writter.add_graph(model, (init_data, init_data) if opt.model == "MCSAT_AV" else init_data)

    for epoch in range(1, opt.epochs + 1):
        time_1 = time.time()
        train_acc, train_loss, cur_lr = train_one_epoch(epoch, train_loader, model, criterion, optimizer, scaler, opt)

        train_time = time.time() - time_1
        total_train_time += train_time
        lr_scheduler.step(train_loss if opt.lr_scheduler == "reduce" else None)

        train_result["accuracy"].append(train_acc)
        train_result["loss"].append(train_loss)
        train_result["lr"].append(cur_lr)

        tb_writter.add_scalar("train_accuracy", train_acc, epoch)
        tb_writter.add_scalar("train_loss", train_loss, epoch)
        tb_writter.add_scalar("train_lr", cur_lr, epoch)

        print(f"the {epoch} epoch, total train time {train_time:.2f}")

        if opt.datasets == "cond1_same_condition":
            # No validation split: do not inspect the held-out test set during
            # training or use it to choose an epoch.
            continue

        selection_loader = val_loader if val_loader is not None else test_loader
        selection_split = "Val" if val_loader is not None else "Test"
        selection_time_start = time.time()
        selection_acc, selection_loss = validate(
            selection_loader, model, criterion, opt, split_name=selection_split)
        selection_time = time.time() - selection_time_start
        if val_loader is not None:
            total_val_time += selection_time
            val_result["accuracy"].append(selection_acc)
            val_result["loss"].append(selection_loss)
            tb_writter.add_scalar("val_accuracy", selection_acc, epoch)
            tb_writter.add_scalar("val_loss", selection_loss, epoch)
        else:
            total_test_time += selection_time
            test_result["accuracy"].append(selection_acc)
            test_result["loss"].append(selection_loss)
            tb_writter.add_scalar("test_accuracy", selection_acc, epoch)
            tb_writter.add_scalar("test_loss", selection_loss, epoch)

        if selection_acc > best_acc:
            best_acc = selection_acc
            best_epoch = epoch
            best_state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                          'best_acc': best_acc, 'selection_split': selection_split}

            print("==> Saving the best model")
            best_save_file = os.path.join(opt.save_folder, f'best_checkpoint_epoch_{best_epoch}.pth')
            torch.save(best_state, best_save_file)

    print('==> Saving the acc, loss and lr during training')
    with open(os.path.join(opt.save_folder, 'train_result.pkl'), "wb") as tf:
        pickle.dump(train_result, tf)

    if opt.datasets == "cond1_same_condition":
        last_state = {'epoch': opt.epochs, 'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(), 'selection_split': 'none'}
        torch.save(last_state, os.path.join(opt.save_folder, 'last_checkpoint.pth'))
    elif val_loader is not None:
        with open(os.path.join(opt.save_folder, "val_result.pkl"), "wb") as vf:
            pickle.dump(val_result, vf)
    else:
        with open(os.path.join(opt.save_folder, "test_result.pkl"), "wb") as tf:
            pickle.dump(test_result, tf)

    print("==> Evaluating model and plotting confusion matrix")
    print(f"Total training time: {total_train_time:.2f} seconds")
    if opt.datasets == "cond1_same_condition":
        print(f"No validation split; using final epoch {opt.epochs}. Evaluating held-out test set once.")
    elif val_loader is not None:
        if best_epoch == 0:
            raise RuntimeError("No best checkpoint was selected from the validation set")
        model.load_state_dict(best_state["model"])
        print(f"Selected checkpoint from validation epoch {best_epoch}; testing held-out test set once.")
    if opt.datasets != "cond1_same_condition":
        print(f"Total validation time: {total_val_time:.2f} seconds")
    test_start = time.time()
    evaluate_and_plot_confusion_matrix(model, test_loader, opt)
    total_test_time += time.time() - test_start
    print(f"Total testing time: {total_test_time:.2f} seconds")


def plot_confusion_matrix(cm, classes, accuracy, normalize=False, cmap=plt.cm.Blues):
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm.astype('float'), row_sums,
                       out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2f}%')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def evaluate_and_plot_confusion_matrix(model, test_loader, opt, threshold=0.8):
    model.eval()
    all_predictions = []
    all_targets = []

    total_TN = 0
    total_FP = 0
    total_FN = 0
    total_TP = 0

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                acoustic, vibration, targets = batch
            else:
                images, targets = batch
            device = next(model.parameters()).device
            targets = targets.to(device)
            if len(batch) == 3:
                outputs = model(acoustic.float().to(device), vibration.float().to(device))
            else:
                outputs = model(images.float().to(device))

            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            probs = torch.softmax(outputs, dim=1)
            prob_normal = probs[:, 0]

            binary_preds = (prob_normal < threshold).long()
            binary_targets = (targets > 0).long()

            total_TN += torch.sum((binary_preds == 0) & (binary_targets == 0)).item()
            total_FP += torch.sum((binary_preds == 1) & (binary_targets == 0)).item()
            total_FN += torch.sum((binary_preds == 0) & (binary_targets == 1)).item()
            total_TP += torch.sum((binary_preds == 1) & (binary_targets == 1)).item()

    cm = confusion_matrix(all_targets, all_predictions, labels=range(opt.num_cls))
    accuracy = 100 * np.trace(cm) / float(np.sum(cm))
    precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)

    FPR = total_FP / (total_FP + total_TN + 1e-8) * 100
    FNR = total_FN / (total_FN + total_TP + 1e-8) * 100

    print('\n' + '=' * 55)
    print(f'🏭 [最终工业部署验证报告] (报警严苛阈值: {threshold})')
    print(f' ➤ L2 检修多分类准确率: {accuracy:.4f}%')
    print(f' ➤ L1 监控误报率 (FPR): {FPR:.4f}%  [ {total_FP} 次误报 / {total_FP + total_TN} 总正常样本 ]')
    print(f' ➤ L1 监控漏报率 (FNR): {FNR:.4f}%  [ {total_FN} 次漏报 / {total_FN + total_TP} 总故障样本 ]')
    print(f' ➤ Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}')
    print('=' * 55 + '\n')

    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(cm, classes=range(opt.num_cls), accuracy=accuracy, normalize=True)

    plt.savefig(os.path.join(opt.save_folder, 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {os.path.join(opt.save_folder, 'confusion_matrix.png')}")


if __name__ == "__main__":
    main()
