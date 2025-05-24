import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import matplotlib.pyplot as plt
from torch_model import get_torch_model
from tf_model import get_tf_model
from utils.utils import get_torch_dataloader, get_tf_dataset

def train_torch(model, train_loader, test_loader, epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        total_batches = len(train_loader)

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress = (batch_idx + 1) / total_batches
            bar_length = 50
            blocks = int(progress * bar_length)
            bar = '█' * blocks + '-' * (bar_length - blocks)
            print(f'Epoch {epoch+1}/{epochs} [{bar}] {progress*100:.1f}%', end='\r')

        epoch_loss = running_loss / total_batches
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss /= len(test_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'\nEpoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    torch.save(model.state_dict(), '../fatou_model.torch')

    # 🎨 Graphe matplotlib
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss', color='orange')
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy (%)', color='red')
    plt.plot(range(1, epochs + 1), val_accuracies, label='Val Accuracy (%)', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('PyTorch Training & Validation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def train_tf(model, train_dataset, test_dataset, epochs=10):
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch + 1
            self.total_batches = len(train_dataset)
            self.batch_count = 0

        def on_batch_end(self, batch, logs=None):
            self.batch_count += 1
            progress = self.batch_count / self.total_batches
            bar_length = 50
            blocks = int(progress * bar_length)
            bar = '█' * blocks + '-' * (bar_length - blocks)
            print(f'Epoch {self.epoch}/{epochs} [{bar}] {progress*100:.1f}%', end='\r')

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        callbacks=[ProgressCallback(), early_stop],
        verbose=0
    )

    model.save('../fatou_model.tensorflow')

    train_losses = history.history['loss']
    val_losses = history.history['val_loss']
    train_accuracies = [acc * 100 for acc in history.history['accuracy']]
    val_accuracies = [acc * 100 for acc in history.history['val_accuracy']]

    # 🎨 Graphe matplotlib
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', color='orange')
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy (%)', color='red')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Accuracy (%)', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('TensorFlow Training & Validation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train'], default='train')
    parser.add_argument('--framework', choices=['torch', 'tf'], required=True)
    args = parser.parse_args()

    train_dir = '../dataset/training'
    test_dir = '../dataset/testing'

    if args.framework == 'torch':
        model = get_torch_model()
        train_loader = get_torch_dataloader(train_dir, train=True)
        test_loader = get_torch_dataloader(test_dir, train=False)
        train_torch(model, train_loader, test_loader)
    else:
        model = get_tf_model()
        train_dataset = get_tf_dataset(train_dir, train=True)
        test_dataset = get_tf_dataset(test_dir, train=False)
        train_tf(model, train_dataset, test_dataset)

if __name__ == '__main__':
    main()
