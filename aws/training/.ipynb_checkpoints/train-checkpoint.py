#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################### IMPORT AND LOGS ########################################################

import argparse
import os
import numpy as np
import xml.etree.ElementTree as ET
import random

import torch
import torchvision
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


from PIL import Image

import logging
logger = logging.getLogger()
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s %(levelname)4s %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.setLevel(logging.DEBUG)


from datetime import datetime
def print_log(message):
    PRINT_LOG = True
    if PRINT_LOG:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f' -> -> -> -> -> -> {current_time}:  {message}')


import sys
print_log(f'python: {sys.version}')
print_log(f'torch: {torch.__version__}')
print_log(f'torchvision: {torchvision.__version__}')

print_log(f'environment: {os.environ}')

#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################### PLATFORM SELECTION #########################################################
try:
    RunningInCOLAB = 'google.colab' in str(get_ipython())
except:
    RunningInCOLAB = False

    
if RunningInCOLAB:
    print_log('Script running in Google Colab')
    defaults_args = dict(
        epochs = 10,
        batch_size = 4,
        learning_rate = 0.005,
        use_cuda = True,

        output_data_dir = "/content/gdrive/My Drive/progetti/_Plansoft/mask_detection/output_data",
        model_dir = "/content/gdrive/My Drive/progetti/_Plansoft/mask_detection/models",
        train = "/content/gdrive/My Drive/progetti/_Plansoft/mask_detection/data/train",
        test = "/content/gdrive/My Drive/progetti/_Plansoft/mask_detection/data/test"
    )
else:
    print_log('Script running in AWS')
    defaults_args = dict(
        epochs = 10,
        batch_size = 4,
        learning_rate = 0.005,
        use_cuda = True,

        output_data_dir = os.environ['SM_OUTPUT_DATA_DIR'],
        model_dir = os.environ['SM_MODEL_DIR'],
        train = os.environ['SM_CHANNEL_TRAINING'],
        test = os.environ['SM_CHANNEL_TEST']
    )
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################### HELPER FUNCTIONS ###########################################################

def decode_xml(xml_path):
    label_to_class = dict(        
        with_mask = 1,
        without_mask = 2,
        mask_weared_incorrect = 3)
    
    root = ET.parse(xml_path).getroot()
    img_filename = root.find('filename').text

    boxes = []
    classes = []
    for obj in root.findall('object'):
        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)

        classes.append(label_to_class[obj.find('name').text])
        boxes.append([xmin, ymin, xmax, ymax])

    return classes, boxes, img_filename


class MaskDataset(object):
    def __init__(self, transforms, data, mode='train'):
        self.transforms = transforms
        self.data = data
        self.idx_with_mask =    [i for i, current_data in enumerate(self.data) if not current_data['without_mask']]
        self.idx_without_mask = [i for i, current_data in enumerate(self.data) if current_data['without_mask']]

        if mode=='train':
            self.minority_class_len = min(len(self.idx_with_mask), len(self.idx_without_mask))
            self.shuffle()
        if mode == 'test':
            self.index_list = self.idx_with_mask + self.idx_without_mask


    def shuffle(self):
        random.shuffle(self.idx_with_mask)
        random.shuffle(self.idx_without_mask)
        self.index_list = self.idx_with_mask[:self.minority_class_len] + self.idx_without_mask[:self.minority_class_len]
        random.shuffle(self.index_list)
        
    def __getitem__(self, index):
        idx = self.index_list[index]
        img = Image.open(self.data[idx]['image_file_path']).convert("RGB")

        ## create target
        boxes = torch.as_tensor(self.data[idx]['boxes'], dtype=torch.float32)
        classes = torch.as_tensor(self.data[idx]['classes'], dtype=torch.int64)
        img_id = torch.tensor([idx])
        target = dict(boxes=boxes, labels=classes, image_id=img_id)

        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.index_list)
    
        
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) # pretrained using COCO
    input_features_number = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features_number, num_classes+1) # replace head
    return model




def get_data(annotation_path, imgs_path):
    print_log(f'parsing xml start')
    data = []
    for label_file_name in os.listdir(annotation_path):

        classes, boxes, img_filename = decode_xml(os.path.join(annotation_path, label_file_name))

        ## skip mask_weared_incorrect
        if 3 in classes:
            continue
            
        data.append(dict(
            image_file_path = os.path.join(imgs_path, img_filename),
            label_file_path = os.path.join(annotation_path, label_file_name),
            classes = classes,
            boxes = boxes,
            without_mask =  2 in classes))
    print_log(f'parsing xml finish')
    return data

#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################### GETTING DATA AND PARAMETERS ################################################
if __name__=='__main__':
    print_log('Main started')
    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=defaults_args['epochs'])
    parser.add_argument('--batch-size', type=int, default=defaults_args['batch_size'])
    parser.add_argument('--learning-rate', type=float, default=defaults_args['learning_rate'])
    parser.add_argument('--use-cuda', type=bool, default=defaults_args['use_cuda'])

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=defaults_args['output_data_dir'])
    parser.add_argument('--model-dir', type=str, default=defaults_args['model_dir'])
    parser.add_argument('--train', type=str, default=defaults_args['train'])
    parser.add_argument('--test', type=str, default=defaults_args['test'])

    args, _ = parser.parse_known_args()

    annotation_path = args.train + "/annotations"
    imgs_path = args.train + "/images"

    # instanciate device (gpu vs cpu)    
    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
    print_log('selected cuda' if args.use_cuda else 'selected gpu')



    # get training data
    training_data = get_data(annotation_path, imgs_path)

    # get model from torch
    model = get_model(num_classes=2)

    # train the model
    model.to(device);


    #getting test data

    test_annotation_path = args.test + "/annotations"
    test_imgs_path = args.test + "/images"
    test_data = get_data(test_annotation_path, test_imgs_path)

    data_transform = transforms.Compose([transforms.ToTensor(),])
    test_dataset = MaskDataset(data_transform, test_data, mode='test')
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, collate_fn=lambda b:tuple(zip(*b)))

#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################### TRAINING LOOP ##############################################################

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    data_transform = transforms.Compose([transforms.ToTensor(),])


    losses_hist = []
    test_epoc_losses_hist = []
    epoc_losses_hist = []

    print_log(f'starting training')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        # shuffle the dataset and balance the classes
        dataset = MaskDataset(data_transform, training_data)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=lambda b:tuple(zip(*b)))

        for imgs, targets in data_loader:

            ## convert cpu tensor to device tensor
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            ## gradient descent step
            optimizer.zero_grad()
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            epoch_loss += losses
            losses_hist.append(losses)



        # model.eval() in faster rcnn we cant compute losses in eval mode
        test_loss = 0
        with torch.no_grad():
            for imgs, targets in test_data_loader:
                ## convert cpu tensor to device tensor
                imgs = list(img.to(device) for img in imgs)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                test_loss_dict = model(imgs, targets)
                test_loss = test_loss + sum(loss for loss in test_loss_dict.values())


        epoc_losses_hist.append(epoch_loss)
        test_epoc_losses_hist.append(test_loss)
        print_log(f'epoch {epoch} finished. Train_losses={epoc_losses_hist[-1]:.3f};  Valid_losses={test_epoc_losses_hist[-1]:.3f};')
        logger.info(f'Train_losses={epoc_losses_hist[-1]:.3f};  Valid_losses={test_epoc_losses_hist[-1]:.3f};')

    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f) # for inference only
    
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################