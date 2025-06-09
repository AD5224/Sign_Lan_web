# Create a virtual environment and install required libraries if not already done
# python -m venv venv
# venv\Scripts\Activate.ps1
# pip install numpy pandas tensorflow opencv-python matplotlib scikit-learn

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
from function import *  # Updated function.py

DATA_PATH = 'MP_Data12'  # Make sure this path is correct

actions = np.array(['1','2','3','4','5','6','7','8','9',
                    'A','B','C','D','E','F','G','H','I',
                    'J','K','L','M','N','O','P','Q','R',
                    'S','T','U','V','W','X','Y','Z','ALL THE BEST','THANK YOU','I LOVE YOU'])
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            try:
                file_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                res = np.load(file_path)
                window.append(res)
            except:
                pass
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(f"X shape: {X.shape}")  # Should show (samples, 30, 126)
print(f"y shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Callbacks
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Model
model = Sequential()
model.add(LSTM(128, return_sequences=True, activation='tanh', input_shape=(30, 126)))  # UPDATED here
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(LSTM(256, return_sequences=True, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(LSTM(128, return_sequences=False, activation='tanh'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train
print("Starting training...")
history = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[tb_callback, early_stopping, reduce_lr], verbose=2)
print("Training completed.")

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_acc}")

# Save
model.save('new_gesture.h5')
model_json = model.to_json()
with open("new_gesture_architecture.json", "w") as json_file:
    json_file.write(model_json)
print("Model saved.")
