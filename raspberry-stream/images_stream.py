import time
import cv2

IMG_REDUCTION = 1
webcam = cv2.VideoCapture(0) #Use camera 0


while True:
    try:
        t0 = time.time()
        (rval, im) = webcam.read()
        im=cv2.flip(im,1,1) #Flip to act as a mirror

        # Resize the image to speed up detection
        mini = cv2.resize(im, (im.shape[1] // IMG_REDUCTION, im.shape[0] // IMG_REDUCTION))

            
        # Show the image
        cv2.imshow('LIVE',   im)
    except:
        print('Error')
        
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop 
    if key == 27: #The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()

