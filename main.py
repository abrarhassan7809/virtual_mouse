import cv2
import numpy as np
import HandTrackingModule as htm
import pyautogui
import time

##############
cam_w, cam_h = 640, 480
##############

cap = cv2.VideoCapture(0)
cap.set(3, cam_w)
cap.set(4, cam_h)

previous_time = 0
detector = htm.handDetector(maxHands=1)

while True:
    # hand land marks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # get tip of the fingers
    # check which finger is up
    # only index finger move
    # convert coordinates
    # values
    # move mouse
    # index and middle finger up
    # find distance b/t them
    # check distance and click the mouse

    # frame rate
    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # display
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
