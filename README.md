<h1 align="center">🖐️ Sign Language Recognition System</h1>
<p align="center">
  <b>Real-time hand gesture recognition using MediaPipe, LSTM, OpenCV & Flask</b><br/>
  <i>Translating hand signs into text and speech for enhanced communication.</i>
</p>

<p align="center">
  <img src="static/images/demo.gif" alt="Demo" width="700"/>
</p>

---

## 🔍 About the Project

This project enables real-time sign language detection using a webcam. Hand gestures are detected using **MediaPipe**, landmarks are passed to an **LSTM model** for prediction, and the output is converted to both **text** and **speech** using `gtts`. The application is wrapped in a lightweight **Flask** web interface for interaction.

---

## 🎬 Demo Video

<video width="100%" controls>
  <source src="https://github.com/AD5224/Sign_Language_Recognition/blob/main/SIGNLANV.mp4?raw=true" type="video/mp4">
  Your browser does not support the video tag.
</video>


## 🛠️ Tech Stack

| Category     | Technologies Used                       |
|--------------|------------------------------------------|
| Frontend     | HTML, CSS, JavaScript                    |
| Backend      | Flask (Python)                           |
| ML Model     | LSTM with Keras/TensorFlow               |
| Vision       | OpenCV, MediaPipe                        |
| Voice Output | gtts (Text-to-Speech)                 |

---

## 🚀 Features

- 🖐️ Real-time hand sign detection
- 🎯 Accurate gesture recognition (A–Z, 1–9)
- 🗣️ Speech output for recognized signs
- 💡 Live webcam preview
- 🌐 Easy-to-use web interface

---

## 📦 Project Structure
```bash

Sign_Language_Recognition/
├── backend/ # Python logic
├── frontend/ # HTML/CSS/JS files
├── static/images/ # Demo media
│ └── demo.gif
├── apptry.py # Main Flask app
├── requirements.txt # Dependencies
├── Procfile # For deployment
└── README.md # Project overview



---

## ▶️ Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/your-username/Sign_Language_Recognition.git
cd Sign_Language_Recognition
2. Install dependencies

pip install -r requirements.txt
3. Run the Flask app

python apptry.py
Open your browser at: http://localhost:5000

🎮 Controls
Key Press	Action
s	Insert space
v	Speak predicted text
q	Quit webcam preview

📈 Future Improvements
Sentence/phrase-level recognition

Voice-to-sign translation

Dynamic gesture detection

Cloud-based deployment
