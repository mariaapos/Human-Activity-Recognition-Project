import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob
from collections import Counter

# Paths
SKELETON_FOLDER = './hrnet_skeletons'
ANNOTATION_FILE = '../labels.csv'
CHECKPOINT_PATH = './bilstm_fc_hrnet/epoch_last.pth'

# Model Hyperparameters
MAX_FRAMES = 300
NUM_KEYPOINTS = 17
NUM_COORDS = 2
INPUT_SIZE = NUM_KEYPOINTS * NUM_COORDS  # 17 * 2 = 34
HIDDEN_SIZE = 256
NUM_LAYERS = 2
NUM_CLASSES = 10 # This will be updated based on the data

# Training Hyperparameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
NUM_EPOCHS = 50

def calculate_accuracy(outputs, labels, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class SkeletonDataset(Dataset):
    """
    Custom Dataset for loading HRNet skeleton data.
    It handles NaN values, padding/truncating, and reshaping.
    """
    def __init__(self, annotation_df, skeleton_folder, max_frames):
        self.df = annotation_df
        self.skeleton_folder = skeleton_folder
        self.max_frames = max_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.skeleton_folder, row['filename'])
        label = row['label_encoded']

        try:
            skeleton_data = np.load(file_path)
            # HRNet Specific: Handle NaN values
            skeleton_data = np.nan_to_num(skeleton_data, nan=0.0)
        except FileNotFoundError:
            return torch.zeros((self.max_frames, INPUT_SIZE)), torch.tensor(-1, dtype=torch.long)

        non_zero_indices = np.where(skeleton_data.any(axis=(1, 2)))[0]
        if len(non_zero_indices) > 0:
            skeleton_data = skeleton_data[non_zero_indices]
        else:
            skeleton_data = np.zeros((1, NUM_KEYPOINTS, NUM_COORDS), dtype=np.float32)

        current_frames = skeleton_data.shape[0]
        if current_frames >= self.max_frames:
            processed_data = skeleton_data[:self.max_frames, :, :]
        else:
            padding_shape = (self.max_frames - current_frames, NUM_KEYPOINTS, NUM_COORDS)
            padding = np.zeros(padding_shape, dtype=np.float32)
            processed_data = np.concatenate((skeleton_data, padding), axis=0)
            
        processed_data = processed_data.reshape(self.max_frames, -1)

        data_tensor = torch.tensor(processed_data, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return data_tensor, label_tensor

# BiLSTM Model Definition
class BiLSTM_FC(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM_FC, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("--- Preparing Data ---")
    skeleton_files = glob.glob(os.path.join(SKELETON_FOLDER, '*.npy'))
    skeleton_info = []
    for path in skeleton_files:
        fname = os.path.basename(path).replace('.npy', '')
        parts = fname.rsplit('_', 2)
        if len(parts) != 3: continue
        youtube_id, time_start, time_end = parts
        try:
            skeleton_info.append({
                'youtube_id': youtube_id.strip(), 'time_start': int(time_start),
                'time_end': int(time_end), 'filename': os.path.basename(path)
            })
        except ValueError: continue
    skeleton_df = pd.DataFrame(skeleton_info)
    print(f"Found and parsed {len(skeleton_df)} skeleton files.")

    if not os.path.exists(ANNOTATION_FILE):
        print(f"Error: Annotation file '{ANNOTATION_FILE}' not found.")
        return
        
    labels_df = pd.read_csv(ANNOTATION_FILE)
    labels_df.columns = [c.strip() for c in labels_df.columns]
    labels_df['youtube_id'] = labels_df['youtube_id'].astype(str).str.strip()
    labels_df['time_start'] = labels_df['time_start'].astype(int)
    labels_df['time_end'] = labels_df['time_end'].astype(int)

    merged_df = pd.merge(skeleton_df, labels_df, on=['youtube_id', 'time_start', 'time_end'], how='inner')
    print(f"Successfully matched {len(merged_df)} skeletons with labels.")

    label_counts = merged_df['label'].value_counts()
    labels_to_keep = label_counts[label_counts > 1].index
    filtered_df = merged_df[merged_df['label'].isin(labels_to_keep)]
    print(f"Filtered out {len(merged_df) - len(filtered_df)} samples from classes with only 1 instance.")

    matched_list = list(zip(filtered_df['filename'], filtered_df['label']))
    if not matched_list:
        print("Error: No skeletons matched after filtering.")
        return

    all_labels = sorted(list(set(label for _, label in matched_list)))
    label_map = {label: i for i, label in enumerate(all_labels)}
    global NUM_CLASSES
    NUM_CLASSES = len(all_labels)
    print(f"Found {NUM_CLASSES} unique classes after initial filtering.")
    
    train_list, temp_list = train_test_split(matched_list, test_size=0.2, random_state=42, stratify=[item[1] for item in matched_list])
    
    temp_labels = [item[1] for item in temp_list]
    temp_label_counts = Counter(temp_labels)
    labels_to_keep_in_temp = {label for label, count in temp_label_counts.items() if count > 1}
    temp_list_filtered = [item for item in temp_list if item[1] in labels_to_keep_in_temp]
    print(f"Filtered out {len(temp_list) - len(temp_list_filtered)} samples from temp list for the val/test split.")

    val_list, test_list = train_test_split(temp_list_filtered, test_size=0.5, random_state=42, stratify=[item[1] for item in temp_list_filtered])
    
    train_df = pd.DataFrame(train_list, columns=['filename', 'label'])
    val_df = pd.DataFrame(val_list, columns=['filename', 'label'])
    test_df = pd.DataFrame(test_list, columns=['filename', 'label'])
    
    train_df['label_encoded'] = train_df['label'].map(label_map)
    val_df['label_encoded'] = val_df['label'].map(label_map)
    test_df['label_encoded'] = test_df['label'].map(label_map)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    train_dataset = SkeletonDataset(train_df, SKELETON_FOLDER, MAX_FRAMES)
    val_dataset = SkeletonDataset(val_df, SKELETON_FOLDER, MAX_FRAMES)
    test_dataset = SkeletonDataset(test_df, SKELETON_FOLDER, MAX_FRAMES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = BiLSTM_FC(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_top1_acc': [], 'val_top1_acc': [],
        'train_top5_acc': [], 'val_top5_acc': []
    }

    print("\n--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, train_top1, train_top5, train_total = 0.0, 0.0, 0.0, 0
        for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            data, labels = data.to(device), labels.to(device)
            if -1 in labels: continue

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            top1_acc, top5_acc = calculate_accuracy(outputs, labels)
            train_top1 += top1_acc.item() * data.size(0)
            train_top5 += top5_acc.item() * data.size(0)
            train_total += data.size(0)

        model.eval()
        val_loss, val_top1, val_top5, val_total = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                data, labels = data.to(device), labels.to(device)
                if -1 in labels: continue

                outputs = model(data)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                top1_acc, top5_acc = calculate_accuracy(outputs, labels)
                val_top1 += top1_acc.item() * data.size(0)
                val_top5 += top5_acc.item() * data.size(0)
                val_total += data.size(0)

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_top1_acc'].append(train_top1 / train_total if train_total > 0 else 0)
        history['train_top5_acc'].append(train_top5 / train_total if train_total > 0 else 0)
        
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_top1_acc'].append(val_top1 / val_total if val_total > 0 else 0)
        history['val_top5_acc'].append(val_top5 / val_total if val_total > 0 else 0)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {history['train_loss'][-1]:.4f}, Top-1: {history['train_top1_acc'][-1]:.2f}%, Top-5: {history['train_top5_acc'][-1]:.2f}% | "
              f"Val Loss: {history['val_loss'][-1]:.4f}, Top-1: {history['val_top1_acc'][-1]:.2f}%, Top-5: {history['val_top5_acc'][-1]:.2f}%")

    torch.save({
        'epoch': NUM_EPOCHS - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, CHECKPOINT_PATH)
    print(f"Model and training history saved to {CHECKPOINT_PATH}")

    print("\n--- Evaluating on Test Set ---")
    model.eval()
    test_top1, test_top5, test_total = 0.0, 0.0, 0
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc="[Test]"):
            data, labels = data.to(device), labels.to(device)
            if -1 in labels: continue
            
            outputs = model(data)
            top1_acc, top5_acc = calculate_accuracy(outputs, labels)
            test_top1 += top1_acc.item() * data.size(0)
            test_top5 += top5_acc.item() * data.size(0)
            test_total += data.size(0)
            
    test_top1_acc = test_top1 / test_total if test_total > 0 else 0
    test_top5_acc = test_top5 / test_total if test_total > 0 else 0
    print(f"Final Test Accuracy -> Top-1: {test_top1_acc:.2f}% | Top-5: {test_top5_acc:.2f}%")

if __name__ == '__main__':
    if not os.path.exists(SKELETON_FOLDER):
        print(f"Error: Skeleton folder '{SKELETON_FOLDER}' not found.")
    elif not os.path.exists(ANNOTATION_FILE):
        print(f"Error: Annotation file '{ANNOTATION_FILE}' not found.")
    else:
        main()