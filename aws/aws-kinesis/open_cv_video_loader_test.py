import numpy as np
import cv2


frame_index = 0
cap = cv2.VideoCapture('sample.mkv')

while(cap.isOpened()):
    ret, frame = cap.read()

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_index += 1
    try:
        cv2.imshow('frame', frame)
    except:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(frame_index)