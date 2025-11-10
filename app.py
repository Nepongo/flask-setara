from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load TFLite models
alphabet_interpreter = tf.lite.Interpreter(model_path="assets/alphabet_model_dnn.tflite")
alphabet_interpreter.allocate_tensors()
vocab_interpreter = tf.lite.Interpreter(model_path="assets/vocabulary_model.tflite")
vocab_interpreter.allocate_tensors()

# Load labels
with open("assets/alphabet_labels.txt") as f:
    alphabet_labels = [line.strip() for line in f]
with open("assets/vocabulary_labels.txt") as f:
    vocab_labels = [line.strip() for line in f]

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    landmarks = np.array(data['landmarks'], dtype=np.float32)
    mode = data.get('mode', 'alphabet')
    if mode == 'alphabet':
        input_details = alphabet_interpreter.get_input_details()
        output_details = alphabet_interpreter.get_output_details()
        alphabet_interpreter.set_tensor(input_details[0]['index'], landmarks.reshape(1, -1))
        alphabet_interpreter.invoke()
        output = alphabet_interpreter.get_tensor(output_details[0]['index'])[0]
        idx = int(np.argmax(output))
        label = alphabet_labels[idx]
        confidence = float(output[idx])
        return jsonify({'text': label, 'confidence': confidence})
    else:
        input_details = vocab_interpreter.get_input_details()
        output_details = vocab_interpreter.get_output_details()
        vocab_interpreter.set_tensor(input_details[0]['index'], landmarks.reshape(1, 30, 126))
        vocab_interpreter.invoke()
        output = vocab_interpreter.get_tensor(output_details[0]['index'])[0]
        idx = int(np.argmax(output))
        label = vocab_labels[idx]
        confidence = float(output[idx])
        return jsonify({'text': label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)