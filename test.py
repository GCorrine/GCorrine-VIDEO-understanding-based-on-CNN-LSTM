import torch
from models.model import CNNLSTM
from utils.data_loader import get_data_loaders
from utils.utils import load_model
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch.amp import autocast  # 使用新的AMP API


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    绘制混淆矩阵
    """
    plt.figure(figsize=(12, 10))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def test(model, device, test_loader, criterion, class_names):
    """
    测试模型性能
    """
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # 使用混合精度评估
            with autocast(device_type='cuda', dtype=torch.float16):
                output = model(data)
                loss = criterion(output, target)

            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(torch.softmax(output, dim=1).cpu().numpy())

    # 计算指标
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f"\nTest Results:")
    print(f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    print(f"Average loss: {test_loss:.4f}")

    # 分类报告
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names))

    # 混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, classes=class_names, normalize=True)

    return accuracy, all_probs


def main():
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # 加载类别名称 (HMDB-51有51个类别)
    try:
        class_names = sorted(os.listdir("data/test"))
        print(f"Found {len(class_names)} classes")
    except FileNotFoundError:
        print("Error: 'data/test' directory not found!")
        return

    # 模型初始化
    model = CNNLSTM(num_classes=len(class_names), hidden_size=512, num_layers=2).to(device)

    # 加载预训练权重
    model_path = "models/best_model.pth"
    try:
        load_model(model, model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # 数据加载
    try:
        _, test_loader = get_data_loaders(
            train_dir="data/train",
            test_dir="data/test",
            batch_size=16,
            seq_length=16
        )
        print(f"Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # 测试模型
    criterion = nn.CrossEntropyLoss()
    accuracy, _ = test(model, device, test_loader, criterion, class_names)
    print(f"\nFinal Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()