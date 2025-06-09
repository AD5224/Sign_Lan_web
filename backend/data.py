from function import *  # Updated function with multi-hand extract_keypoints
from time import sleep
import cv2
import os

# Create folders if not exist
for action in actions:
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# Initialize Mediapipe Hands model
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):

                # Read image from saved frames
                frame_path = f'Image/{action}/{sequence}.png'
                frame = cv2.imread(frame_path)

                if frame is None:
                    print(f"Frame not found: {frame_path}")
                    continue

                # Detect hand landmarks
                image, results = mediapipe_detection(frame, hands)

                # Draw landmarks (optional, just for visual debugging)
                draw_styled_landmarks(image, results)

                # Show the frame (optional)
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting for {action}, Video {sequence}', (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500)  # Pause at start
                else:
                    cv2.putText(image, f'Collecting for {action}, Video {sequence}', (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                # Extract keypoints (both hands now)
                keypoints = extract_keypoints(results)
                
                # Save keypoints
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Exit if 'q' is pressed
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
