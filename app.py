import os
import numpy as np
import gdown
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Limit memory growth
physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


app = Flask(__name__, template_folder=".")

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained ML model
MODEL_URL = "https://drive.google.com/uc?export=download&id=1x2ip-V9LRgy9lWvvRofbTlRdMYMLeOZN"
MODEL_PATH = "model.keras"  # Change this to your model file

# Download the model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)


# Load the model
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    """Loads an image, preprocesses it, and makes a prediction using the model."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the class index

    class_labels = ["Cat", "Dog", "Other"]  # Change based on your model classes
    return class_labels[predicted_class]

@app.route("/", methods=["GET", "POST"])
def upload_and_predict():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]
        if file.filename == "":
            return "No selected file"

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Make a prediction
            prediction = predict_image(file_path)

            return f"Prediction: {prediction} <br><img src='/static/uploads/{filename}' width='300'>"

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
