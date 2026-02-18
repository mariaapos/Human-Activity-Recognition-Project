import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob
from collections import Counter
import matplotlib.pyplot as plt

# Configuration for Pixie
# Paths
SKELETON_FOLDER = 'pixie_skeletons'
ANNOTATION_FILE = 'labels.csv'
CHECKPOINT_PATH = './bilstm_fc_pixie/epoch_last.pth'

# Model Hyperparameters
MAX_FRAMES = 300
NUM_KEYPOINTS = 145
NUM_COORDS = 3
INPUT_SIZE = NUM_KEYPOINTS * NUM_COORDS
HIDDEN_SIZE = 256
NUM_LAYERS = 2
NUM_CLASSES = 10 # This will be updated based on the data

# Testing Hyperparameters
BATCH_SIZE = 32

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
        return torch.tensor(processed_data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

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

def plot_history(history):
    plt.figure(figsize=(18, 5))

    # Loss vs. Epochs
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs (Pixie)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Top-1 Accuracy vs. Epochs
    plt.subplot(1, 3, 2)
    plt.plot(history['train_top1_acc'], label='Train Top-1 Acc')
    plt.plot(history['val_top1_acc'], label='Validation Top-1 Acc')
    plt.title('Top-1 Accuracy vs. Epochs (Pixie)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Top-5 Accuracy vs. Epochs
    plt.subplot(1, 3, 3)
    plt.plot(history['train_top5_acc'], label='Train Top-5 Acc')
    plt.plot(history['val_top5_acc'], label='Validation Top-5 Acc')
    plt.title('Top-5 Accuracy vs. Epochs (Pixie)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

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

    labels_df = pd.read_csv(ANNOTATION_FILE)
    labels_df.columns = [c.strip() for c in labels_df.columns]
    labels_df['youtube_id'] = labels_df['youtube_id'].astype(str).str.strip()
    labels_df['time_start'] = labels_df['time_start'].astype(int)
    labels_df['time_end'] = labels_df['time_end'].astype(int)

    merged_df = pd.merge(skeleton_df, labels_df, on=['youtube_id', 'time_start', 'time_end'], how='inner')
    
    label_counts = merged_df['label'].value_counts()
    labels_to_keep = label_counts[label_counts > 1].index
    filtered_df = merged_df[merged_df['label'].isin(labels_to_keep)]

    matched_list = list(zip(filtered_df['filename'], filtered_df['label']))
    
    all_labels = sorted(list(set(label for _, label in matched_list)))
    label_map = {label: i for i, label in enumerate(all_labels)}
    
    global NUM_CLASSES
    NUM_CLASSES = len(all_labels)
    print(f"Re-created dataset with {NUM_CLASSES} classes.")
    
    _, temp_list = train_test_split(matched_list, test_size=0.2, random_state=42, stratify=[item[1] for item in matched_list])
    
    temp_labels = [item[1] for item in temp_list]
    temp_label_counts = Counter(temp_labels)
    labels_to_keep_in_temp = {label for label, count in temp_label_counts.items() if count > 1}
    temp_list_filtered = [item for item in temp_list if item[1] in labels_to_keep_in_temp]

    _, test_list = train_test_split(temp_list_filtered, test_size=0.5, random_state=42, stratify=[item[1] for item in temp_list_filtered])
    
    test_df = pd.DataFrame(test_list, columns=['filename', 'label'])
    test_df['label_encoded'] = test_df['label'].map(label_map)

    print(f"Loaded {len(test_df)} samples for testing.")

    test_dataset = SkeletonDataset(test_df, SKELETON_FOLDER, MAX_FRAMES)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file not found at '{CHECKPOINT_PATH}'")
        return

    print(f"Loading model from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH)
    
    model = BiLSTM_FC(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    history = checkpoint['history']
    
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
    
    print("\n--- Test Results ---")
    print(f"Top-1 Accuracy: {test_top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {test_top5_acc:.2f}%")
    
    print("\n--- Plotting Training History ---")
    plot_history(history)

if __name__ == '__main__':
    if not os.path.exists(SKELETON_FOLDER) or not os.path.exists(ANNOTATION_FILE):
        print(f"Error: Make sure '{SKELETON_FOLDER}' and '{ANNOTATION_FILE}' exist.")
    else:
        main()