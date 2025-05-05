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
from function import *  # your helper file

app = Flask(__name__, template_folder='../frontend', static_folder='../frontend/assets')
app.secret_key = '9a8d9b1dcd314758b9487db5692492b8'

# --- Load model ---
with open('model/gesture60.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('model/gesture_model_60frames.h5')

# --- Mediapipe setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(False, 2, 0, 0.5, 0.5)
mp_drawing = mp.solutions.drawing_utils

actions = list("123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [
    "ALL THE BEST", "THANK YOU", "I LOVE YOU"
]

translator = Translator()
USERS_FILE = 'users.json'

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        json.dump([], f)

number_to_word = {
    '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four', '5': 'Five',
    '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine', '0': 'Zero'
}

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise Exception("Could not open camera")

# --- Globals ---
sequence = []
sentence = []
latest_frame = None
lock = threading.Lock()

threshold = 0.75
stable_counter = 0
stable_action = None
freeze_until = 0
current_locked_action = None

# --- Helper functions ---
def is_r_pose(results):
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            idx_tip = hand.landmark[8]
            mid_tip = hand.landmark[12]
            dist = np.sqrt((idx_tip.x - mid_tip.x)**2 + (idx_tip.y - mid_tip.y)**2)
            if dist < 0.05:
                return True
    return False

def mediapipe_detection(image, hands_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands_model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

# --- Frame capture thread ---
def capture_frames():
    global latest_frame, sequence, sentence, stable_counter, stable_action
    global freeze_until, current_locked_action

    while True:
        success, frame = camera.read()
        if not success:
            continue

        x1, y1, x2, y2 = 0, 40, 300, 400
        roi = frame[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(frame, (15, 15), 30)
        blurred[y1:y2, x1:x2] = roi
        frame = cv2.rectangle(blurred, (x1, y1), (x2, y2), (255, 255, 255), 2)

        image, results = mediapipe_detection(roi, hands)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    blurred[y1:y2, x1:x2], hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec((0, 255, 0), 2, 2),
                    mp_drawing.DrawingSpec((255, 0, 0), 2)
                )

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-60:]

        if len(sequence) == 60:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predicted_index = np.argmax(res)
            predicted_action = actions[predicted_index]

            # Fix confusing pairs
            prob_L = res[actions.index('L')]
            prob_9 = res[actions.index('9')]
            if prob_L > prob_9 and (prob_L - prob_9) > 0.10:
                predicted_action = 'L'
            elif prob_9 > prob_L and (prob_9 - prob_L) > 0.10:
                predicted_action = '9'

            prob_P = res[actions.index('P')]
            prob_T = res[actions.index('T')]
            if prob_T > prob_P and (prob_T - prob_P) > 0.10:
                predicted_action = 'T'

            if predicted_action == '2' and is_r_pose(results):
                predicted_action = 'R'

            if time.time() < freeze_until:
                predicted_action = current_locked_action

            if not current_locked_action:
                if res[predicted_index] > threshold:
                    if predicted_action == stable_action:
                        stable_counter += 1
                    else:
                        stable_action = predicted_action
                        stable_counter = 1

                    if stable_counter >= 6:
                        if len(sentence) == 0 or stable_action != sentence[-1]:
                            sentence.append(stable_action)

                        if stable_action in ['R', 'T', 'L', '9', 'P']:
                            freeze_until = time.time() + 2.0
                            current_locked_action = stable_action
                            sequence.clear()

                        stable_counter = 0
                        stable_action = None
            else:
                if predicted_action != current_locked_action and res[predicted_index] > 0.80:
                    current_locked_action = None

        if len(sentence) > 10:
            sentence = sentence[-10:]

        cv2.putText(blurred, ''.join(sentence), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        with lock:
            latest_frame = blurred.copy()

        time.sleep(0.025)

# --- Start thread ---
t = threading.Thread(target=capture_frames)
t.daemon = True
t.start()

# --- Frame generator ---
def generate_frames():
    global latest_frame
    while True:
        with lock:
            if latest_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            if not ret:
                continue
            frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.025)

# --- Routes ---
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
        text = ''.join(sentence)
        translated_hi = translator.translate(text, dest='hi').text
        translated_mr = translator.translate(text, dest='mr').text
        return jsonify({
            'english': text,
            'hindi': translated_hi,
            'marathi': translated_mr
        })
    return jsonify({'error': 'No prediction yet'})

@app.route('/speak/<lang>', methods=['GET'])
def speak(lang):
    if sentence:
        text = ''.join(sentence)
        for num, word in number_to_word.items():
            text = text.replace(num, word)
        try:
            if lang == 'hindi':
                tts = gTTS(text=translator.translate(text, dest='hi').text, lang='hi', slow=False)
            elif lang == 'marathi':
                tts = gTTS(text=translator.translate(text, dest='mr').text, lang='mr', slow=False)
            else:
                tts = gTTS(text=text, lang='en', slow=False)
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

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    sentence.clear()
    return jsonify({'status': 'sentence cleared'})

# --- Run ---
if __name__ == '__main__':
    app.run(debug=True, threaded=True, use_reloader=False)
