import time
import cv2
import base64
import requests



class mylog(object):
    def __init__(self, log_level=5):
        self.log_level = log_level

    def log(self, text, log_level=5):
        if log_level >= self.log_level:
            print(text)
logger = mylog(5)




def invoke_endpoint(img, index):
    print(img.shape)
    

    retval, buff = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buff).decode()


    aws_url = 'https://9vhqydl7l4.execute-api.eu-west-1.amazonaws.com/beta/predict'

    t0 = time.time()
    r = requests.post(aws_url, json={'image_data': jpg_as_text})
    t_tot = time.time() - t0
    logger.log(f'Processing time = {t_tot:.3f}', log_level=5)

    print(r.text)


    plot_result(img, r.text, index)


def plot_result(img, log, index):
    import matplotlib.pyplot as plt
    from matplotlib import patches
    import json

    json_contents = json.loads(log)

    fig,ax = plt.subplots(1)
    ax.imshow(img[...,::-1])

    json_content = json_contents[0]

    for box, score, label in zip(json_content["boxes"], json_content["scores"], json_content["labels"]):
        if (score < 0.97) and (score>0.03):
            continue

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
    plt.savefig(f'boxed-frames/out_{index:03d}.jpg')



frame_index = 0
img_index = 0
cap = cv2.VideoCapture('sample.mkv')

while(cap.isOpened()):
    ret, frame = cap.read()

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_index += 1


    cv2.imshow('frame', frame)
    if not frame_index%5:
        invoke_endpoint(frame, img_index)
        img_index += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(frame_index, img_index)