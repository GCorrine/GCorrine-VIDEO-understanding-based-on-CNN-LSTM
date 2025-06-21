import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, seq_length=16, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.seq_length = seq_length
        self.train = train
        self.video_files = []
        self.labels = []
        self.class_to_idx = {}

        # 遍历每个类别文件夹
        for label_class, action_dir in enumerate(sorted(os.listdir(root_dir))):
            action_path = os.path.join(root_dir, action_dir)
            if os.path.isdir(action_path):
                self.class_to_idx[action_dir] = label_class
                # 遍历每个视频文件
                for video_file in sorted(os.listdir(action_path)):
                    if video_file.endswith(('.mp4', '.avi', '.mov')):
                        self.video_files.append(os.path.join(action_path, video_file))
                        self.labels.append(label_class)


    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        frames = self.load_frames(video_path)

        # 确保frames是numpy数组的列表
        if isinstance(frames, np.ndarray):
            frames = [frames[i] for i in range(frames.shape[0])]

        transformed_frames = []
        for frame in frames:
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)

            if self.transform:
                frame = self.transform(frame)
            transformed_frames.append(frame)

        frames_tensor = torch.stack(transformed_frames)
        label = self.labels[idx]

        return frames_tensor, label

    def load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 训练时随机采样，测试时均匀采样
        if self.train:
            frame_indices = sorted(np.random.choice(
                range(total_frames),
                size=min(total_frames, self.seq_length),
                replace=False
            ))
        else:
            frame_interval = max(1, total_frames // self.seq_length)
            frame_indices = range(0, total_frames, frame_interval)[:self.seq_length]

        for i in range(max(frame_indices) + 1):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()

        # 如果帧数不足，进行填充
        while len(frames) < self.seq_length:
            frames.append(frames[-1])

        return frames


def get_data_loaders(train_dir, test_dir, batch_size=32, seq_length=16):
    # 添加详细路径检查
    print(f"Checking train dir: {train_dir}")
    print(f"Checking test dir: {test_dir}")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # 打印目录内容
    print(f"Train dir contents: {os.listdir(train_dir)[:5]}...")
    print(f"Test dir contents: {os.listdir(test_dir)[:5]}...")

    # 训练数据增强
    train_transform = transforms.Compose([
        transforms.ToPILImage(),  # 将numpy数组转换为PIL Image
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 测试数据转换
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    try:
        train_dataset = VideoDataset(train_dir, transform=train_transform,
                                     seq_length=seq_length, train=True)
        test_dataset = VideoDataset(test_dir, transform=test_transform,
                                    seq_length=seq_length, train=False)

        print(f"Found {len(train_dataset)} training samples")
        print(f"Found {len(test_dataset)} test samples")

        if len(test_dataset) == 0:
            raise RuntimeError("Test dataset is empty! Please check your test directory structure")

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=2, pin_memory=True)

        return train_loader, test_loader

    except Exception as e:
        print(f"Error creating datasets: {str(e)}")
        raise


def collate_fn(batch):
    frames, labels = list(zip(*batch))
    frames = torch.stack(frames, 0)
    labels = torch.tensor(labels, dtype=torch.long)
    return frames, labels