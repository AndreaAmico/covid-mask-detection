import time
import cv2
import boto3
import base64
import requests




class mylog(object):
    def __init__(self, log_level=5):
        self.log_level = log_level

    def log(self, text, log_level=5):
        if log_level >= self.log_level:
            print(text)
logger = mylog(5)




def invoke_endpoint(img):
    t0 = time.time()

    retval, buff = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buff).decode()


    aws_url = 'https://9vhqydl7l4.execute-api.eu-west-1.amazonaws.com/beta/predict'
    r = requests.post(aws_url, json={'image_data': jpg_as_text})

    print(r.text)

    t_tot = time.time() - t0
    logger.log(f'Processing time = {t_tot:.3f}', log_level=5)






IMG_REDUCTION = 1
webcam = cv2.VideoCapture(0) #Use camera 0


while True:
    try:
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
    if key == 32: # spacebar
        invoke_endpoint(im)

    

# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()

