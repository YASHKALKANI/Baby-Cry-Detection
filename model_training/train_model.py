import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Dataset path
DATA_PATH = "/home/bm-13/Desktop/Yash/Ex/Baby-cry-pr/archive/Baby Crying Sounds"

# 🎵 Feature extraction (MFCC)
def extract_features(file_path, max_len=40):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        pad_width = max_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
        return mfcc
    except Exception as e:
        print(f"Error with {file_path}: {e}")
        return None

# Load dataset
features, labels = [], []
for label in os.listdir(DATA_PATH):
    folder = os.path.join(DATA_PATH, label)
    if not os.path.isdir(folder):
        continue
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            data = extract_features(os.path.join(folder, file))
            if data is not None:
                features.append(data)
                labels.append(label)

X = np.array(features)
y = np.array(labels)

# 🔢 Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train = torch.tensor(X_train).unsqueeze(1).float()  # (batch, 1, 40, 40)
X_test = torch.tensor(X_test).unsqueeze(1).float()
y_train = torch.tensor(y_train).long()
y_test = torch.tensor(y_test).long()

# Define CNN model
class CryCNN(nn.Module):
    def __init__(self, num_classes):
        super(CryCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = CryCNN(num_classes=len(np.unique(y_encoded)))

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20
train_loss_history = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    train_loss_history.append(loss.item())
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "baby_cry_model_cpu.pth")
print("✅ Model saved as baby_cry_model_cpu.pth")

# Plot loss
plt.plot(train_loss_history)
plt.title("Training Loss (CPU Model)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
