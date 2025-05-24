# utils/utils.py
import torch
from torchvision import datasets, transforms
import tensorflow as tf
from PIL import Image
import numpy as np

# Chargeur de données PyTorch
def get_torch_dataloader(data_dir, batch_size=32, train=True):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
        transforms.RandomRotation(10) if train else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader

# Chargeur de données TensorFlow
def get_tf_dataset(data_dir, batch_size=32, train=True):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=(64, 64),
        batch_size=batch_size,
        shuffle=train
    )
    if train:
        dataset = dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
        dataset = dataset.map(lambda x, y: (tf.image.random_brightness(x, 0.2), y))
    dataset = dataset.map(lambda x, y: (x / 255.0, y))  # Normalisation
    return dataset

# Prétraitement d'une image pour la prédiction
def preprocess_image(image_path, framework='torch'):
    img = Image.open(image_path).convert('RGB')
    if framework == 'torch':
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img = transform(img).unsqueeze(0)  # Ajouter une dimension de batch
        return img
    else:  # TensorFlow
        img = img.resize((64, 64))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)  # Ajouter une dimension de batch
        return img

# Noms des classes
def get_class_names():
    return ['Glioma', 'Meningioma', 'Notumor', 'Pituitary']