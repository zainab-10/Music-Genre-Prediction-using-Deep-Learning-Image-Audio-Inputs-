from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import librosa
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename  # Import secure_filename

app = Flask(__name__)

SAMPLE_RATE = 5000
# Load the saved model
model = load_model('InceptionV3_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if both image and audio files are provided
        if 'image' not in request.files or 'audio' not in request.files:
            return jsonify({'error': 'Both image and audio files are required'})

        image_file = request.files['image']
        audio_file = request.files['audio']

        # Use secure_filename to secure the file names
        image_filename = secure_filename(image_file.filename)
        audio_filename = secure_filename(audio_file.filename)

        # Save the files to a temporary folder (adjust the path accordingly)
        image_path = f"temp/{image_filename}"
        audio_path = f"temp/{audio_filename}"

        image_file.save(image_path)
        audio_file.save(audio_path)

        # Assuming the uploaded file is an image
        img_array = image.img_to_array(image.load_img(image_path, target_size=(224, 224)))
        img_array = img_array/255.0

        # Assuming the uploaded file is an audio file
        audio_data, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        # Your audio preprocessing code here

        # Make predictions
        prediction = model.predict([np.array([img_array]), np.array([audio_data])])

        # Get the class label
        predicted_class = np.argmax(prediction)

        # Remove the temporary files if needed
        # os.remove(image_path)
        # os.remove(audio_path)

        return render_template('index.html', class_name=list(class_mapping.keys())[predicted_class])

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
