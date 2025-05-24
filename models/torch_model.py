import torch
import torch.nn as nn
import torch.nn.functional as F

class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Entrée : 64x64x3, Sortie : 64x64x32
        self.pool = nn.MaxPool2d(2, 2)  # Sortie : 32x32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Sortie : 32x32x64
        self.pool2 = nn.MaxPool2d(2, 2)  # Sortie : 16x16x64
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Aplatir : 64 * 16 * 16 = 16384
        self.fc2 = nn.Linear(128, 4)  # 4 classes : glioma, meningioma, notumor, pituitary

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Aplatir
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_torch_model():
    return BrainTumorCNN()