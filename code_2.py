import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize hand tracking module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize screen width and height for pyautogui
screen_width, screen_height = pyautogui.size()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip horizontally for mirror effect

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Calculate Euclidean distance between index finger and middle finger tips
            distance = math.sqrt((index_finger_tip.x - middle_finger_tip.x) ** 2 + (index_finger_tip.y - middle_finger_tip.y) ** 2)

            if distance < 0.02:  # Adjust the threshold as needed
                pyautogui.click()
                cv2.putText(img, "Click", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            else:
                # Move mouse using index finger position
                x, y = int(index_finger_tip.x * screen_width), int(index_finger_tip.y * screen_height)
                pyautogui.moveTo(x, y)

            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
