import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
vb   []
labels = []

for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_dir):
        continue

    for img_path in os.listdir(class_dir):
        data_aux = []

        img = cv2.imread(os.path.join(class_dir, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.extend([landmark.x, landmark.y])

                print("Hand landmarks shape:", np.asarray(hand_data).shape)  # Print shape for debugging

                data_aux.append(hand_data)

            data.append(data_aux)
            labels.append(int(dir_))

print('Dataset was Successfully Created!')

# Convert data to NumPy array
data = np.asarray(data)

# Save the data and labels into a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print('Data was Successfully Saved to "data.pickle"!')
