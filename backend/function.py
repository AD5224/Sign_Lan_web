import cv2
import numpy as np
import os
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

def extract_keypoints(results):
    lh = np.zeros(21*3)  # 63 points
    rh = np.zeros(21*3)  # 63 points

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()

            if handedness == 'Left':
                lh = keypoints
            elif handedness == 'Right':
                rh = keypoints

    return np.concatenate([lh, rh])  # 126 features now

# Path for exported data
DATA_PATH = os.path.join('MP_Data12')

actions = np.array(['THANK YOU','I LOVE YOU','ALL THE BEST'])

no_sequences = 60
sequence_length = 60


# actions = np.array(['1','2','3','4','5','6','7','8','9',
#                     'A','B','C','D','E','F','G','H','I',
#                     'J','K','L','M','N','O','P','Q','R',
#                     'S','T','U','V','W','X','Y','Z'])
