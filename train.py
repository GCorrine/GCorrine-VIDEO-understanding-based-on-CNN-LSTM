import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from models.model import CNNLSTM
from utils.data_loader import get_data_loaders
import torch.nn as nn
import time
from utils.utils import save_model, plot_lr_curve

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        """
        Args:
            patience (int): 验证集性能未改善的等待epoch数
            delta (float): 被视为改善的最小变化量
            verbose (bool): 是否打印早停信息
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """保存当前最佳模型"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def train(model, device, train_loader, optimizer, criterion, epoch, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()

        # 使用新版的autocast API
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 50 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}\t'
                  f'Time: {time.time() - start_time:.2f}s')

    train_loss = total_loss / len(train_loader)
    train_acc = 100. * correct / total
    print(f'Train set: Average loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
    return train_loss, train_acc


def validate(model, device, test_loader, criterion):
    if test_loader is None or len(test_loader) == 0:
        print("Warning: Test loader is empty!")
        return float('inf'), 0.0  # 返回最差情况

    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    try:
        with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        val_loss = total_loss / len(test_loader)
        val_acc = 100. * correct / total
        print(f'Validation set: Average loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%\n')
        return val_loss, val_acc

    except Exception as e:
        print(f"Validation error: {str(e)}")
        return float('inf'), 0.0

def main():
    # 初始化设置
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 启用性能优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    # 模型初始化
    model = CNNLSTM(num_classes=51, hidden_size=512, num_layers=2)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Using device: {device}")

    # 优化器设置
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scaler = GradScaler()  # 新API
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss().to(device)

    # 初始化早停
    early_stopping = EarlyStopping(patience=5, verbose=True)

    # 数据加载
    train_loader, test_loader = get_data_loaders(
        train_dir="data/train",
        test_dir="data/test",
        batch_size=8,  # 减小batch size以适应GPU显存
        seq_length=16
    )

    # 初始化学习率记录列表
    lr_history = []

    # 训练循环
    best_acc = 0
    for epoch in range(1, 31):
        print(f"\n=== Epoch {epoch} ===")

        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch, scaler)
        val_loss, val_acc = validate(model, device, test_loader, criterion)

        scheduler.step(val_loss)

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        print(f"Current learning rate: {current_lr:.2e}")

        # 早停检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            # 训练结束时绘制学习率曲线
            plot_lr_curve(lr_history, save_path='results/lr_curve.png')
            break

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            model_to_save = model.module if hasattr(model, 'module') else model
            save_model(model_to_save, "models/best_model.pth")
            print(f"New best model saved with accuracy: {best_acc:.2f}%")

    # 训练正常结束后也绘制学习率曲线
    plot_lr_curve(lr_history, 'lr_curve.png')


        # # 定期保存检查点
        # if epoch % 5 == 0:
        #     model_to_save = model.module if hasattr(model, 'module') else model
        #     save_model(model_to_save, f"models/epoch_{epoch}.pth")


if __name__ == "__main__":
    main()