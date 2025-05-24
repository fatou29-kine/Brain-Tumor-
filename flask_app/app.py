# flask_app/app.py
import sys
import os
import uuid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, render_template
import torch
import tensorflow as tf
from models.torch_model import get_torch_model
from models.tf_model import get_tf_model
from utils.predict import predict_image

app = Flask(__name__)

# Load the models
try:
    torch_model = get_torch_model()
    torch_model.load_state_dict(torch.load('../fatou_model.torch', map_location='cpu'))
    torch_model.eval()
except Exception as e:
    print(f"Error loading PyTorch model: {e}")
    torch_model = None

try:
    tf_model = tf.keras.models.load_model('../fatou_model.tensorflow')
except Exception as e:
    print(f"Error loading TensorFlow model: {e}")
    tf_model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None
    if request.method == 'POST':
        framework = request.form['model']
        file = request.files['image']
        if file and framework in ['pytorch', 'tensorflow']:
            if framework == 'pytorch' and torch_model is None:
                prediction = "Erreur : Le modèle PyTorch n'a pas pu être chargé."
            elif framework == 'tensorflow' and tf_model is None:
                prediction = "Erreur : Le modèle TensorFlow n'a pas pu être chargé."
            else:
                # Sauvegarder l'image uploadée
                filename = f"{uuid.uuid4().hex}_{file.filename}"
                upload_folder = os.path.join('static', 'uploads')
                os.makedirs(upload_folder, exist_ok=True)
                file_path = os.path.join(upload_folder, filename)
                file.save(file_path)

                # Préparer l'URL pour affichage
                image_url = '/' + file_path

                # Prédiction
                if framework == 'pytorch':
                    prediction = predict_image('torch', file_path, torch_model)
                else:
                    prediction = predict_image('tensorflow', file_path, tf_model)

        else:
            prediction = "Erreur : Veuillez sélectionner un modèle et une image valide."
    return render_template('index.html', prediction=prediction, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
