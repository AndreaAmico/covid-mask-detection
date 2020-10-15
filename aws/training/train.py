import argparse
import os
import numpy as np
import xml.etree.ElementTree as ET
import random
import time

import torch
import torchvision
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from PIL import Image

def decode_xml(xml_path):
    label_to_class = dict(
        without_mask = 0,
        with_mask = 1,
        mask_weared_incorrect = 2)
    
    root = ET.parse(xml_path).getroot()

    boxes = []
    classes = []
    for obj in root.findall('object'):
        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)

        classes.append(label_to_class[obj.find('name').text])
        boxes.append([xmin, ymin, xmax, ymax])

    return classes, boxes


class MaskDataset(object):
    def __init__(self, transforms, training_data):
        self.transforms = transforms
        self.training_data = training_data

        self.idx_with_mask =    [i for i, current_data in enumerate(self.training_data) if not current_data['without_mask']]
        self.idx_without_mask = [i for i, current_data in enumerate(self.training_data) if current_data['without_mask']]
        self.minority_class_len = min(len(self.idx_with_mask), len(self.idx_without_mask))
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.idx_with_mask)
        random.shuffle(self.idx_without_mask)
        self.index_list = self.idx_with_mask[:self.minority_class_len] + self.idx_without_mask[:self.minority_class_len]
        random.shuffle(self.index_list)
        
    def __getitem__(self, index):
        idx = self.index_list[index]
        img = Image.open(self.training_data[idx]['image_file_path']).convert("RGB")

        ## create target
        boxes = torch.as_tensor(self.training_data[idx]['boxes'], dtype=torch.float32)
        classes = torch.as_tensor(self.training_data[idx]['classes'], dtype=torch.int64)
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
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features_number, num_classes) # replace head
    return model


if __name__ =='__main__':
    t0 = time.time()
    print('Main started')
    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.005)
    parser.add_argument('--use-cuda', type=bool, default=False)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

    args, _ = parser.parse_known_args()
    
    MODEL_PATH = args.model_dir
    ANNOTATION_PATH = args.train + "/annotations"
    IMG_PATH = args.train + "/images"

    
    print(f'parsing xml - {time.time()-t0:.1f}')
    ## parse xml annotations
    NUMBER_OF_IMAGES = len(os.listdir(ANNOTATION_PATH))
    data_train = {}
    for idx in range(NUMBER_OF_IMAGES):
        image_file_name = f'maksssksksss{idx}.png'
        label_file_name = f'maksssksksss{idx}.xml'

        classes, boxes = decode_xml(os.path.join(ANNOTATION_PATH, label_file_name))

        ## skip mask_weared_incorrect
        if 2 in classes:
            continue
            
        data_train[idx] = dict(
            image_file_path = os.path.join(IMG_PATH, image_file_name),
            label_file_path = os.path.join(ANNOTATION_PATH, label_file_name),
            classes = classes,
            boxes = boxes,
            without_mask = 0 in classes)
    data_train = [data_train[k] for k in data_train]
        
    # instanciate device (gpu vs cpu)
    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
    
    # get model from torch
    model = get_model(num_classes = 2)
    
    # train the model
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    data_transform = transforms.Compose([transforms.ToTensor(),])

    losses_hist = []
    epoc_losses_hist = []
    
    print(f'starting training - {time.time()-t0:.1f}')
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        # shuffle the dataset and balance the classes
        dataset = MaskDataset(data_transform, data_train)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=lambda b:tuple(zip(*b)))
        a = 0
        
        for imgs, targets in data_loader:
            
            print(f'start training on batch {a}/{len(data_loader)} - {time.time()-t0:.1f}')
            
        
            ## convert cpu tensor to device tensor
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            ## gradient descent step
            optimizer.zero_grad()
            loss_dict = model([imgs[0]], [targets[0]])
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            epoch_loss += losses
            losses_hist.append(losses)
            
            print(f'finished training on batch {a}/{len(data_loader)} - {time.time()-t0:.1f}')
            a = a + 1
            
        epoc_losses_hist.append(epoch_loss)
        print(f'epoch {epoch} finished - {time.time()-t0:.1f}')
        
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f) # for inference only
        print(f'model saved - {time.time()-t0:.1f}')