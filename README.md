# Optical Character Recognition (Handwritten Digit Recognition)

An interactive web-based Optical Character Recognition (OCR) application powered by a pre-trained CNN model exported to ONNX (`handwriting_model.onnx`).

Draw digits (0-9) directly on your browser screen using your mouse or touch input and get real-time digit predictions!

---

## Try the app [here](https://ocr-web-chi-silk.vercel.app)

## Try running it locally(Web App)

1. **Run the startup script**:
   ```bash
   ./run.sh
   ```

2. **Open in Browser**:
   Navigate to **[http://localhost:5001](http://localhost:5001)** or **[http://127.0.0.1:5001](http://127.0.0.1:5001)** in your browser.

*(Note: We use port **5001** because port 5000 is used by macOS AirPlay Receiver).*

---

## Features

- **Interactive Drawing Canvas**: Draw using mouse or touch input with adjustable brush sizes.
- **Real-Time Auto-Prediction**: Instantly updates predictions as you draw on the canvas.
- **Model Preview**: View the 28x28 preprocessed input image after centering, cropping, and blurring to see exactly what the model processes.
- **Probability Distribution**: Detailed confidence breakdown across all digits (0–9).
- **Pre-trained ONNX Model**: Utilizes `handwriting_model.onnx` with OpenCV image processing matching standard MNIST normalization.

---

## Project Structure

- `app.py`: Flask web server & OpenCV preprocessing backend.
- `templates/index.html`: Responsive HTML5 drawing UI & model prediction dashboard.
- `handwriting_model.onnx`: Pre-trained ONNX model file.
- `ModelTrainer.py`: Keras/TensorFlow CNN training script.
- `ToONNX.py`: Conversion utility from SavedModel to ONNX format.
- `main.cpp`: OpenCV C++ desktop application (legacy terminal tool).
- `run.sh`: Bash launcher script.
