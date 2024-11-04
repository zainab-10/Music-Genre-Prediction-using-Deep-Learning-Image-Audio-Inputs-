# Music Genre Prediction using Deep Learning (Image & Audio Inputs)

This project demonstrates a music genre prediction system that leverages deep learning by combining both **album cover images** and **audio samples**. The web-based application, built with **Flask**, allows users to upload an image and audio file, which are processed to predict the genre using a **fine-tuned InceptionV3 model** and **Librosa** for audio feature extraction.

![Output Example](https://github.com/zainab-10/Music-Genre-Prediction-using-Deep-Learning-Image-Audio-Inputs-/blob/main/Screenshot%202024-10-07%20235948.png)  <!-- Replace with your actual image URL -->

## Features
- Multi-modal learning: Uses both image and audio inputs for genre prediction.
- Image processing with **InceptionV3** (TensorFlow/Keras).
- Audio feature extraction with **Librosa**.
- Web interface powered by **Flask** for file uploads and real-time predictions.
- Secure file handling with **Werkzeug**.

## Technologies
- **TensorFlow/Keras** for deep learning.
- **Librosa** for audio processing.
- **Flask** for the web app interface.
- **Python** for backend logic and processing.

## How to Use
1. Clone the repository.
2. Install the required dependencies.
3. Run the Flask app and upload your album cover and audio file to get the genre prediction.
