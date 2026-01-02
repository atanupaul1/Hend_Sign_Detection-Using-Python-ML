import pickle
import cv2
import mediapipe as mp
import numpy as np
import threading
from queue import Queue
import tkinter as tk
from PIL import Image, ImageTk

# Function to handle video capturing and processing
def process_frames(queue):
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            print('Error capturing video')
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]
                data_aux = [(x - min(x_), y - min(y_)) for x, y in zip(x_, y_)]

                H, W, _ = frame.shape
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                data_aux = np.asarray(data_aux)
                data_aux = data_aux.reshape(1, -1)  # Reshape to (1, number_of_features)
                prediction = model.predict(data_aux)
                predicted_character = chr(ord('A') + int(prediction[0]))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

        queue.put(frame)

    cap.release()
    cv2.destroyAllWindows()

def update_gui():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)

        hand_sign_label.config(image=img)
        hand_sign_label.image = img

    root.quit()

def stop_processing():
    frame_queue.put(None)

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Hand Sign Detection")

    frame_queue = Queue()

    video_frame = tk.LabelFrame(root, text="Video Stream")
    video_frame.grid(row=0, column=0, padx=10, pady=10)

    hand_sign_label = tk.Label(video_frame)
    hand_sign_label.grid(row=0, column=0, padx=5, pady=5)

    stop_button = tk.Button(root, text="Stop", command=stop_processing)
    stop_button.grid(row=1, column=0, pady=5)

    process_thread = threading.Thread(target=process_frames, args=(frame_queue,))
    process_thread.start()

    gui_thread = threading.Thread(target=update_gui)
    gui_thread.start()

    root.mainloop()

    process_thread.join()
    gui_thread.join()
