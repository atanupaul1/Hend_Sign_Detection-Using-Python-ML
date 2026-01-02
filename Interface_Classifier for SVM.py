import pickle
import cv2
import mediapipe as mp
import numpy as np
import threading
from sklearn.svm import SVC
import tkinter as tk

# Function to handle video capturing and processing
def process_frames():
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                   12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
                   23: 'X', 24: 'Y', 25: 'Z'}

    cap = cv2.VideoCapture(0)

    root = tk.Tk()
    root.title("Hand Sign Detection")
    label = tk.Label(root, font=('Helvetica', 36))
    label.pack()

    while True:
        ret, frame = cap.read()

        if not ret:
            print('Error capturing video')
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                x_ = []
                y_ = []
                data_aux = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                H, W, _ = frame.shape
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                decision_values = model.decision_function([np.asarray(data_aux)])
                max_decision_index = np.argmax(decision_values)
                accuracy = min(decision_values[0][max_decision_index], 1.0) * 100

                print(f'Recognized character: {predicted_character}')
                print(f'Accuracy: {accuracy:.2f}%')

                label.config(text=f'Recognized character: {predicted_character}\nAccuracy: {accuracy:.2f}%')

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

        cv2.imshow('Hand Sign Detection', frame)

        if cv2.waitKey(10) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    root.mainloop()

# Main function to start the video processing thread
def main():
    try:
        thread = threading.Thread(target=process_frames)
        thread.start()
    except Exception as e:
        print('Error:', str(e))

if __name__ == '__main__':
    main()
