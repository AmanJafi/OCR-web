import base64
import io
import re
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load pre-trained ONNX model
MODEL_PATH = "handwriting_model.onnx"
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Loaded model '{MODEL_PATH}' successfully.")
except Exception as e:
    print(f"Error loading ONNX model '{MODEL_PATH}': {e}")
    session = None


def decode_base64_image(data_url):
    """Decode base64 image data URL to single-channel (grayscale) numpy array."""
    image_data = re.sub('^data:image/.+;base64,', '', data_url)
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # Handle transparent background or RGBA/RGB channels
    if img is None:
        return None

    if len(img.shape) == 3 and img.shape[2] == 4:
        # Extract alpha channel if available or convert to gray
        alpha = img[:, :, 3]
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        # Combine alpha and grayscale: non-transparent pixels become white drawing
        gray = cv2.bitwise_and(gray, gray, mask=alpha)
    elif len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    return gray


def preprocess_for_mnist(img_gray):
    """
    Preprocess drawing canvas image into MNIST format (28x28, centered, 0..1 float)
    matching the exact preprocessing steps from C++ main.cpp.
    """
    if img_gray is None or np.max(img_gray) == 0:
        empty_input = np.zeros((1, 28, 28, 1), dtype=np.float32)
        empty_preview = np.zeros((28, 28), dtype=np.uint8)
        return empty_input, empty_preview

    # Binary thresholding
    _, work = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)

    # Find external contours
    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        empty_input = np.zeros((1, 28, 28, 1), dtype=np.float32)
        empty_preview = np.zeros((28, 28), dtype=np.uint8)
        return empty_input, empty_preview

    # Find contour with largest area
    best_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    x, y, w, h = cv2.boundingRect(contours[best_idx])

    # Crop to square while maintaining aspect ratio
    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)
    offset_x = (size - w) // 2
    offset_y = (size - h) // 2
    square[offset_y:offset_y + h, offset_x:offset_x + w] = work[y:y + h, x:x + w]

    # Resize square to 20x20
    resized = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)

    # Add 4-pixel border on all sides to make it 28x28
    final_img = cv2.copyMakeBorder(resized, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)

    # Shift center of mass to center of image (14, 14)
    m = cv2.moments(final_img, True)
    if m['m00'] != 0:
        cx = m['m10'] / m['m00']
        cy = m['m01'] / m['m00']
        shift_x = 14.0 - cx
        shift_y = 14.0 - cy
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        final_img = cv2.warpAffine(final_img, M, (28, 28))

    # Apply 3x3 Gaussian Blur
    blurred = cv2.GaussianBlur(final_img, (3, 3), 0.5)

    # Normalize to 0.0 - 1.0 range
    input_tensor = (blurred.astype(np.float32) / 255.0).reshape(1, 28, 28, 1)
    preview_img = blurred.astype(np.uint8)

    return input_tensor, preview_img


def encode_preview_base64(img_uint8):
    """Convert 28x28 uint8 array into base64 PNG image string for web preview."""
    # Scale up for crisp display in browser
    scaled = cv2.resize(img_uint8, (112, 112), interpolation=cv2.INTER_NEAREST)
    _, buffer = cv2.imencode('.png', scaled)
    return "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if session is None:
        return jsonify({'error': 'ONNX model is not loaded'}), 500

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    img_gray = decode_base64_image(data['image'])
    input_tensor, preview_img = preprocess_for_mnist(img_gray)

    # Check if canvas was empty
    if np.max(input_tensor) == 0:
        return jsonify({
            'empty': True,
            'prediction': None,
            'confidence': 0.0,
            'probabilities': [0.0] * 10,
            'preview': encode_preview_base64(preview_img)
        })

    # Run ONNX model inference
    outputs = session.run([output_name], {input_name: input_tensor})[0]
    probabilities = outputs[0].tolist()

    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    return jsonify({
        'empty': False,
        'prediction': predicted_digit,
        'confidence': round(confidence * 100, 2),
        'probabilities': [round(p * 100, 2) for p in probabilities],
        'preview': encode_preview_base64(preview_img)
    })


if __name__ == '__main__':
    print("Starting Optical Character Recognition Web Server on http://0.0.0.0:5001 (or http://localhost:5001)")
    app.run(host='0.0.0.0', port=5001, debug=True)
