import torch
import os
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def plot_lr_curve(lr_history, save_path='lr_curve.png'):
    """
    绘制学习率变化曲线
    Args:
        lr_history (list): 记录的学习率列表
        save_path (str): 图片保存路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lr_history, 'b', linewidth=2)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)

    # 标记关键点
    changes = np.where(np.diff(lr_history) != 0)[0] + 1
    for change in changes:
        plt.axvline(x=change, color='r', linestyle='--', alpha=0.3)
        plt.text(change, lr_history[change],
                 f'{lr_history[change]:.2e}',
                 ha='center', va='bottom')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_model(model, filename):
    """
    保存模型的函数。

    参数:
    - model: 要保存的模型。
    - filename: 保存模型的文件名，包括路径。
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # 保存模型的状态字典
    torch.save(model.state_dict(), filename)
    logging.info(f"模型已保存到 {filename}")


def load_model(model, filename):
    """
    加载模型的函数。

    参数:
    - model: 要加载的模型。
    - filename: 加载模型的文件名，包括路径。
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"模型文件 {filename} 不存在")

    # 加载模型的状态字典
    state_dict = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    logging.info(f"模型已从 {filename} 加载")


def setup_logging(log_dir="logs"):
    """
    配置日志记录的函数。
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def count_parameters(model):
    """
    计算模型参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed=42):
    """
    设置随机种子以保证可重复性
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False