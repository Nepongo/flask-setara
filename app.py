from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from flask_cors import CORS
import os  # <-- TAMBAHAN: Dibutuhkan untuk membangun file path yang aman

app = Flask(__name__)
CORS(app)

# --- REVISI DI SINI ---
# Tentukan 'BASE_DIR' (folder tempat app.py ini berada)
# Ini adalah cara paling aman untuk menemukan file di server produksi.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Buat path absolut ke file aset Anda
alphabet_model_path = os.path.join(BASE_DIR, 'assets', 'alphabet_model_dnn.tflite')
vocab_model_path = os.path.join(BASE_DIR, 'assets', 'vocabulary_model.tflite')
alphabet_labels_path = os.path.join(BASE_DIR, 'assets', 'alphabet_labels.txt')
vocab_labels_path = os.path.join(BASE_DIR, 'assets', 'vocabulary_labels.txt')
# --- AKHIR REVISI ---

@app.route('/')
def home():
    # Ini sudah benar
    return render_template('index.html')

# Load TFLite models
# Menggunakan path absolut yang baru
alphabet_interpreter = tf.lite.Interpreter(model_path=alphabet_model_path)
alphabet_interpreter.allocate_tensors()
vocab_interpreter = tf.lite.Interpreter(model_path=vocab_model_path)
vocab_interpreter.allocate_tensors()

# Load labels
# Menggunakan path absolut yang baru
with open(alphabet_labels_path) as f:
    alphabet_labels = [line.strip() for line in f]
with open(vocab_labels_path) as f:
    vocab_labels = [line.strip() for line in f]

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    landmarks = np.array(data['landmarks'], dtype=np.float32)
    mode = data.get('mode', 'alphabet')
    
    try:
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
            # Pastikan data 'landmarks' memiliki shape yang bisa di-reshape ke (1, 30, 126)
            vocab_interpreter.set_tensor(input_details[0]['index'], landmarks.reshape(1, 30, 126))
            vocab_interpreter.invoke()
            output = vocab_interpreter.get_tensor(output_details[0]['index'])[0]
            idx = int(np.argmax(output))
            label = vocab_labels[idx]
            confidence = float(output[idx])
            return jsonify({'text': label, 'confidence': confidence})
    
    except Exception as e:
        # Menambahkan error handling jika terjadi masalah saat prediksi
        # Ini akan membantu Anda melihat error di log Railway jika ada
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Baris ini tidak akan dijalankan oleh Gunicorn, tapi bagus untuk testing lokal
    app.run(host='0.0.0.0', port=5000)