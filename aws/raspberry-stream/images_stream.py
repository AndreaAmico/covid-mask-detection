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
    print(img.shape)
    

    retval, buff = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buff).decode()


    aws_url = 'https://9vhqydl7l4.execute-api.eu-west-1.amazonaws.com/beta/predict'

    t0 = time.time()
    r = requests.post(aws_url, json={'image_data': jpg_as_text})
    t_tot = time.time() - t0
    logger.log(f'Processing time = {t_tot:.3f}', log_level=5)

    print(r.text)


    plot_result(img, r.text)


def plot_result(img, log):
    import matplotlib.pyplot as plt
    from matplotlib import patches
    import json

    json_contents = json.loads(log)

    fig,ax = plt.subplots(1)
    ax.imshow(img)

    json_content = json_contents[0]

    for box, score, label in zip(json_content["boxes"], json_content["scores"], json_content["labels"]):
        lab_to_color = {
            1:(0,1,0),
            2:(1,0,0)
        }
        
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),
                                linewidth=1,
                                edgecolor=lab_to_color[int(label)],
                                facecolor=lab_to_color[int(label)]+(0.2,))
        ax.add_patch(rect)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig('out.png')





IMG_REDUCTION = 2
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
        invoke_endpoint(mini)

    

# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()

