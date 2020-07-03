# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 21:37:20 2020

@author: dhruv
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:39:45 2020

@author: dhruv
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:15:33 2020

@author: dhruv
"""

# this makes the dataset from the helmet forlder and divides it into train and test


from mrcnn.config import Config
from mrcnn.model import MaskRCNN


	

class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "helmet_cfg"

	NUM_CLASSES = 1 + 2
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model_path = 'final_weights.h5'
model.load_weights(model_path, by_name=True)



import cv2
import numpy as np
class_names = [
    'BG', 'helmet','head']
#from visualize_cv2 import model, display_instances, class_names
def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


colors = random_colors(len(class_names))

class_dict = {
    name: color for name, color in zip(class_names, colors)
}

class_dict = {'BG': (255,255,255),
 'helmet': (0,255,0),
 'head': (0,0,255)}


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):

    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue
        
        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        if label == 'helmet':
            color = class_dict[label]
            score = scores[i] if scores is not None else None
            caption = '{} {:.2f}'.format(label, score) if score else label
            mask = masks[:, :, i]

            image = apply_mask(image, mask, color)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            image = cv2.putText(image, "Helmet" , (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        if label == 'head' :
            color = class_dict[label]
            score = scores[i] if scores is not None else None
            caption = '{} {:.2f}'.format(label, score) if score else label
            mask = masks[:, :, i]
            image = apply_mask(image, mask, color)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            image = cv2.putText(image, "No Helmet", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
       

    return image


capture = cv2.VideoCapture('hats_demo.mp4') # name of video file to be played

# or VideoCapture(0) for webcam 
size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('hats_demo_5000.avi', codec, 60.0, size) # name to save the video file

while(capture.isOpened()):
    ret, frame = capture.read()
    if ret:
        # add mask to frame
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], 
                            class_names,r['scores'])
        
        output.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    


capture.release()
output.release()
cv2.destroyAllWindows()

# to test on an image
img = cv2.imread('4pep.png')
results = model.detect([img], verbose=0)
r = results[0]
frame = display_instances(img, r['rois'], r['masks'], r['class_ids'],class_names,r['scores'])
cv2.imshow('frame', img)
