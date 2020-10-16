import json
import requests
import torch
import torchvision
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import os

import logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)4s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)



def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) # pretrained using COCO
    input_features_number = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features_number, num_classes) # replace head
    return model

def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading the model.')    
    
    model = get_model(2)
    if torch.cuda.is_available():
        with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
            model.load_state_dict(torch.load(f))
    else:
        with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
            model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))

    model.to(device).eval()
    logger.info('Done loading model')
    return model


def input_fn(request_body, content_type='application/json'):
    logger.info('Deserializing the input data.')
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        url = input_data['url']
        logger.info(f'Image url: {url}')
        
        image_data = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        transfs = transforms.Compose([transforms.ToTensor(),])
        image_data = transfs(image_data)
        return image_data
    raise Exception(f'Requested unsupported ContentType in content_type {content_type}')

    
def predict_fn(input_data, model):
    logger.info('Generating prediction based on input parameters.')
    if torch.cuda.is_available():
        input_data = input_data.cuda()
    else:
        input_data = input_data

    input_data = input_data.squeeze()
    input_data = input_data.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        out = model(input_data)
        
    return out

def output_fn(prediction_output, accept='application/json'):
    logger.info('Serializing the generated output.')

    output = []

    if torch.cuda.is_available():
        for out in prediction_output:
            out = {k:out[k].detach().cpu().tolist() for k in out}
            output.append(out)
    else:
        for out in prediction_output:
            out = {k:out[k].tolist() for k in out}
            output.append(out)
    if accept == 'application/json':
        return json.dumps(output), accept
    raise Exception(f'Requested unsupported ContentType in Accept:{accept}')