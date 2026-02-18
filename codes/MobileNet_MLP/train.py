import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm
from torchvision.models import (
    resnet18, ResNet18_Weights,
    vgg16, VGG16_Weights,
    mobilenet_v2, MobileNet_V2_Weights
)
from torchvision.models.video import r3d_18, R3D_18_Weights

# Collect video file info
video_files = glob.glob(os.path.join('../kinetics_subset/videos', '*.mp4'))
video_info = []
failed_files = []
for path in video_files:
    fname = os.path.basename(path).replace('.mp4', '')
    parts = fname.rsplit('_', 2)
    if len(parts) != 3:
        failed_files.append((path, "Invalid filename format"))
        continue
    youtube_id, time_start, time_end = parts
    try:
        video_info.append({
            'youtube_id': youtube_id.strip(),
            'time_start': int(time_start),
            'time_end': int(time_end),
            'path': path
        })
    except ValueError as e:
        failed_files.append((path, f"Error parsing time: {e}"))
        continue

video_df = pd.DataFrame(video_info)
print(f"Parsed {len(video_df)} video files.")
if failed_files:
    print(f"Failed to parse {len(failed_files)} files:")
    for path, reason in failed_files[:5]:
        print(f"  - {path}: {reason}")

# Load and merge labels
labels_df = pd.read_csv('../labels.csv')
labels_df.columns = [c.strip() for c in labels_df.columns]
labels_df['youtube_id'] = labels_df['youtube_id'].astype(str).str.strip()
labels_df['time_start'] = labels_df['time_start'].astype(int)
labels_df['time_end'] = labels_df['time_end'].astype(int)
filtered = pd.merge(
    video_df, labels_df,
    on=['youtube_id','time_start','time_end'],
    how='left'
)
unmatched = filtered[filtered['label'].isna()]
if not unmatched.empty:
    print(f"{len(unmatched)} videos could not be matched to labels:")
    print(unmatched[['youtube_id','time_start','time_end','path']].head())
print(f"Filtered to {len(filtered)} videos, {filtered['label'].notna().sum()} with labels.")
matched_list = [(row.path, row.label) for row in filtered.itertuples() if pd.notna(row.label)]

# Split dataset
train_list, temp_list = train_test_split(matched_list, test_size=0.2, random_state=42)
val_list, test_list = train_test_split(temp_list, test_size=0.5, random_state=42)
print(f"Train: {len(train_list)}, Validation: {len(val_list)}, Test: {len(test_list)}")
label_set = {lbl: idx for idx, lbl in enumerate(sorted(set(label for _, label in matched_list)))}

# Compute class weights
labels_only = [label for _, label in matched_list]
label_counts = Counter(labels_only)
sorted_labels = sorted(label_set)
total = sum(label_counts.values())
weights = [total / label_counts.get(cls, 1) for cls in sorted_labels]
weights = torch.tensor(weights, dtype=torch.float32)
weights = weights / weights.sum()

# Dataset with frame-drop stride
class KineticsStridedDataset(Dataset):
    def __init__(self, video_label_list, label_set=None, frame_stride=4, transform=None):
        self.video_label_list = video_label_list
        self.stride = frame_stride
        self.transform = transform
        labels = [lbl for _, lbl in video_label_list]
        self.label_to_idx = {l: i for i, l in enumerate(sorted(set(labels)))} if label_set is None else label_set

    def __len__(self):
        return len(self.video_label_list)

    def __getitem__(self, idx):
        path, label_str = self.video_label_list[idx]
        label = self.label_to_idx[label_str]
        cap = cv2.VideoCapture(path)
        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.stride == 0:
                frame = cv2.resize(frame, (224, 224))
                tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                if self.transform:
                    tensor = self.transform(tensor)
                frames.append(tensor)
            frame_idx += 1
        cap.release()
        if len(frames) == 0:
            frames = [torch.zeros((3,224,224))]
        video_tensor = torch.stack(frames)
        return video_tensor, label

# Collate function to pad variable lengths
def pad_collate(batch):
    videos, labels = zip(*batch)
    lengths = [v.shape[0] for v in videos]
    max_len = max(lengths)
    padded = []
    for v in videos:
        if v.shape[0] < max_len:
            pad = torch.zeros((max_len - v.shape[0], *v.shape[1:]), dtype=v.dtype)
            v = torch.cat([v, pad], dim=0)
        padded.append(v)
    videos_tensor = torch.stack(padded)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return videos_tensor, labels_tensor, torch.tensor(lengths, dtype=torch.long)

class ClassifierHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, num_classes=100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.fc(x)
    
class MobileNetMLPActivityModel(nn.Module):
    def __init__(self, num_classes, hidden_dim=512):
        super().__init__()
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.features = mobilenet.features
        self.pool     = nn.AdaptiveAvgPool2d((1,1))
        feat_dim      = 1280

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, feat_dim)
        )
        self.classifier = ClassifierHead(in_dim=feat_dim,
                                         num_classes=num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        fmap   = self.features(x)
        pooled = self.pool(fmap).flatten(start_dim=1)   # [B*T, 1280]
        seq    = pooled.view(B, T, -1)                  # [B, T, 1280]
        vid_f  = seq.mean(dim=1)                        # temporal average
        enc    = self.mlp(vid_f)                        # [B, 1280]
        return self.classifier(enc)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(label_set)
model = MobileNetMLPActivityModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
total_epochs = 40

# Training and validation loops
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = top1_correct = top5_correct = total = 0
    loop = tqdm(loader, desc="Training", leave=True)
    for videos, labels, lengths in loop:
        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total += labels.size(0)
        _, top1_pred = outputs.topk(1, dim=1)
        top1_correct += top1_pred.squeeze().eq(labels).sum().item()
        _, top5_pred = outputs.topk(5, dim=1)
        for i in range(labels.size(0)):
            if labels[i] in top5_pred[i]: top5_correct += 1
        loop.set_postfix({'Loss': f"{loss.item():.4f}",
                          'Top-1': f"{100*top1_correct/total:.2f}%",
                          'Top-5': f"{100*top5_correct/total:.2f}%"})
    return total_loss/len(loader), 100*top1_correct/total, 100*top5_correct/total

def validate_model(model, loader, criterion, device):
    model.eval()
    total_loss = top1_correct = top5_correct = total = 0
    loop = tqdm(loader, desc="Validating", leave=True)
    with torch.no_grad():
        for videos, labels, lengths in loop:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total += labels.size(0)
            _, top1_pred = outputs.topk(1, dim=1)
            top1_correct += top1_pred.squeeze().eq(labels).sum().item()
            _, top5_pred = outputs.topk(5, dim=1)
            for i in range(labels.size(0)):
                if labels[i] in top5_pred[i]: top5_correct += 1
            loop.set_postfix({'Loss': f"{loss.item():.4f}",
                              'Top-1': f"{100*top1_correct/total:.2f}%",
                              'Top-5': f"{100*top5_correct/total:.2f}%"})
    return total_loss/len(loader), 100*top1_correct/total, 100*top5_correct/total

# DataLoaders and training run
train_ds = KineticsStridedDataset(train_list, label_set=label_set, frame_stride=4)
val_ds   = KineticsStridedDataset(val_list,   label_set=label_set, frame_stride=4)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=5, pin_memory=True, collate_fn=pad_collate)
val_loader   = DataLoader(val_ds, batch_size=8, num_workers=5, pin_memory=True, collate_fn=pad_collate)

history = {'train_loss':[], 'val_loss':[], 'train_top1':[], 'val_top1':[],
           'train_top5':[], 'val_top5':[]}

os.makedirs("mobilenet_mlp_checkpoints", exist_ok=True)

for epoch in range(total_epochs):
    t_loss, t1, t5 = train_one_epoch(model, train_loader, optimizer, criterion, device)
    v_loss, v1, v5 = validate_model(model, val_loader, criterion, device)
    history['train_loss'].append(t_loss)
    history['train_top1'].append(t1)
    history['train_top5'].append(t5)
    history['val_loss'].append(v_loss)
    history['val_top1'].append(v1)
    history['val_top5'].append(v5)
    print(f"Epoch {epoch+1}/{total_epochs}: "
          f"Train L={t_loss:.4f},1={t1:.2f}%,5={t5:.2f}% | "
          f"Val L={v_loss:.4f},1={v1:.2f}%,5={v5:.2f}%")

    ckpt = {
        "epoch": epoch + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
    }

    ckpt_path = f"./mobilenet_mlp_checkpoints/ckpt_epoch_{epoch+1:02d}.pth"
    torch.save(ckpt, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")