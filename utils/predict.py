# utils/predict.py

import sys
import os

# 🔧 Ajouter le chemin parent pour que les modules soient trouvés
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import tensorflow as tf
from models.torch_model import get_torch_model
from models.tf_model import get_tf_model
from utils.utils import preprocess_image, get_class_names

def predict_image(framework, image_path, model):
    classes = get_class_names()
    image = preprocess_image(image_path, framework=framework)
    
    if framework == 'torch':
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            return classes[predicted.item()]
    else:  # tensorflow
        output = model.predict(image)
        predicted = output.argmax(axis=1)[0]
        return classes[predicted]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', choices=['torch', 'tf'], required=True)
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()

    # Load the appropriate model based on framework
    if args.framework == 'torch':
        model = get_torch_model()
        model.load_state_dict(torch.load('fatou_model.torch', map_location='cpu'))  # 🚨 ajusté le chemin si le fichier est dans la racine
        model.eval()
    else:
        model = tf.keras.models.load_model('fatou_model.tensorflow')  # 🚨 idem ici

    prediction = predict_image(args.framework, args.image, model)
    print(f'Classe prédite : {prediction}')

if __name__ == '__main__':
    main()
