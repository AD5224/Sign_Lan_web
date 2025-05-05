from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, session, send_file
import cv2
import threading
import time
import json
import numpy as np
import mediapipe as mp
from keras.models import model_from_json
from googletrans import Translator
from gtts import gTTS
from io import BytesIO
import os
from function import *  # Your helper

# Initialize Flask app
app = Flask(
    __name__,
    template_folder='../frontend',
    static_folder='../frontend/assets'
)
app.secret_key = '9a8d9b1dcd314758b9487db5692492b8'

# Load model
with open('model/sign_language_model_architecture.json', 'r') as f:
    model_json = f.read()
model = model_from_json(model_json)
model.load_weights('model/sign_language_high_acc_model.h5')

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

actions = list("123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
translator = Translator()
USERS_FILE = 'users.json'

char_to_word = {
    'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E',
    'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J',
    'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O',
    'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T',
    'U': 'You', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z',
    '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four', '5': 'Five',
    '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine', '0': 'Zero'
}

def get_full_sentence(char_list):
    return ' '.join([char_to_word.get(c, c) for c in char_list])

# Create users.json if missing
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        json.dump([], f)

# Camera Setup
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise Exception("Could not open video device")

# Global Variables
sequence = []
sentence = []
predictions = []
latest_frame = None
lock = threading.Lock()
threshold = 0.75  # Lowered slightly for better A, R detection

# Mediapipe Detection
def mediapipe_detection(image, hands_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands_model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

# Capture Frames
def capture_frames():
    global latest_frame, sequence, sentence, predictions
    while True:
        success, frame = camera.read()
        if not success:
            continue

        cropframe = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)

        image, results = mediapipe_detection(cropframe, hands)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-60:]

        if len(sequence) == 60:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))

            if len(predictions) >= 10:
                last_preds = predictions[-10:]
                most_common = max(set(last_preds), key=last_preds.count)
                if most_common == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) == 0 or actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])

        if len(sentence) > 10:
            sentence = sentence[-10:]

        # Draw predicted sentence
        cv2.putText(frame, ''.join(sentence), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        with lock:
            latest_frame = frame.copy()

        time.sleep(0.03)

# Start capture thread
t = threading.Thread(target=capture_frames)
t.daemon = True
t.start()

# Stream frames for browser
def generate_frames():
    global latest_frame
    while True:
        with lock:
            if latest_frame is None:
                time.sleep(0.05)
                continue
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            if not ret:
                continue
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

# -------------- Routes ----------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        with open(USERS_FILE, 'r') as f:
            users = json.load(f)

        if any(u['username'] == username for u in users):
            return "Username already exists!"

        users.append({'username': username, 'password': password})
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f)

        return redirect(url_for('home'))

    return render_template('signup.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    with open(USERS_FILE, 'r') as f:
        users = json.load(f)

    for user in users:
        if user['username'] == username and user['password'] == password:
            session['user'] = username
            return redirect(url_for('dashboard'))

    return "Invalid credentials!"

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('home'))
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['GET'])
def predict():
    if sentence:
        latest_text = ''.join(sentence)
        full_sentence = get_full_sentence(latest_text)

        translated_hi = translator.translate(full_sentence, dest='hi').text
        translated_mr = translator.translate(full_sentence, dest='mr').text

        return jsonify({
            'english': full_sentence,
            'hindi': translated_hi,
            'marathi': translated_mr
        })

    return jsonify({'error': 'No prediction yet'})

@app.route('/speak/<lang>', methods=['GET'])
def speak(lang):
    if sentence:
        text = ''.join(sentence)
        full_sentence = get_full_sentence(text)

        try:
            if lang == 'hindi':
                translated_text = translator.translate(full_sentence, dest='hi').text
                tts = gTTS(text=translated_text, lang='hi')
            elif lang == 'marathi':
                translated_text = translator.translate(full_sentence, dest='mr').text
                tts = gTTS(text=translated_text, lang='mr')
            else:
                tts = gTTS(text=full_sentence, lang='en')

            mp3_fp = BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            return send_file(mp3_fp, mimetype='audio/mpeg')

        except Exception as e:
            print(f"Speech error: {e}")
            return "Speech generation error", 500

    return "No sentence available", 400

@app.route('/add_space', methods=['POST'])
def add_space():
    sentence.append(' ')
    return jsonify({'status': 'space added'})

# Run App
if __name__ == '__main__':
    app.run(debug=True, threaded=True, use_reloader=False)
