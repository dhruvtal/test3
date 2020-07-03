# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 23:04:56 2020

@author: dhruv
"""

from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray

 
class helmetDataset(Dataset):
  # load the dataset definitions
  def load_dataset(self, dataset_dir, is_train=True):
    # define two class, can add more classes, just add the index number 
    self.add_class("dataset", 1, "helmet") #Change required
    self.add_class("dataset", 2, "head") #Change required
    # define data locations, all images of classes must be in a single folder
	# named images and all annotations must be in annots can be changed accordingly though
    images_dir = dataset_dir + '/images/'
    annotations_dir = dataset_dir + '/annots/'
    # find all images

    for filename in listdir(images_dir):
      # extract image id
      image_id = filename[:-4]
      #print(‘IMAGE ID: ‘,image_id)
      # skip all images after 80 if we are building the train set
      if is_train and int(image_id) >= 80: #set limit for your train and test set
        continue
      # skip all images before 80 if we are building the test/val set
      if not is_train and int(image_id) < 81:
        continue
      img_path = images_dir + filename
      ann_path = annotations_dir + image_id + '.xml'
      # add to dataset
      self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids = [0,1,2]) # for your case it is 0:BG, 1:PerWithHel.., 2:PersonWithoutHel… #Change required

# extract bounding boxes from an annotation file
  def extract_boxes(self, filename):
    # load and parse the file
    tree = ElementTree.parse(filename)
    # get the root of the document
    root = tree.getroot()
    # extract each bounding box
    boxes = list()
    #for box in root.findall('.//bndbox'):
    for box in root.findall('.//object'):
      name = box.find('name').text #Change required
      xmin = int(box.find('./bndbox/xmin').text)
      ymin = int(box.find('./bndbox/ymin').text)
      xmax = int(box.find('./bndbox/xmax').text)
      ymax = int(box.find('./bndbox/ymax').text)
      #coors = [xmin, ymin, xmax, ymax, name]
      coors = [xmin, ymin, xmax, ymax, name] #Change required
      boxes.append(coors)
# extract image dimensions
    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)
    return boxes, width, height

# load the masks for an image
  def load_mask(self, image_id):
    # get details of image
    info = self.image_info[image_id]
    # define box file location
    path = info['annotation']
    # load XML
    boxes, w, h = self.extract_boxes(path)
    # create one array for all masks, each on a different channel
    masks = zeros([h, w, len(boxes)], dtype='uint8')
    # create masks
    class_ids = list()
    for i in range(len(boxes)):
      box = boxes[i]
      row_s, row_e = box[1], box[3]
      col_s, col_e = box[0], box[2]
      if (box[4] == 'helmet'):#Change required #change this to your .XML file
        masks[row_s:row_e, col_s:col_e, i] = 1 #Change required #assign number to your class_id
        class_ids.append(self.class_names.index('helmet')) #Change required
      else:
        masks[row_s:row_e, col_s:col_e, i] = 2 #Change required
        class_ids.append(self.class_names.index('head')) #Change required

    return masks, asarray(class_ids, dtype='int32')

    # load an image reference
  def image_reference(self, image_id):
    info = self.image_info[image_id]
    return info['path']

# train set
train_set = helmetDataset()
train_set.load_dataset('helmet', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
 

# test/val set
test_set = helmetDataset()
test_set.load_dataset('helmet', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))



# define a configuration for the model
class helmetConfig(Config):
    # define the name of the configuration
    NAME = "helmet_cfg"
# number of classes (background + personWithoutHelmet + personWithHelmet)
    NUM_CLASSES = 1 + 2 #Change required
# number of training steps per epoch
    STEPS_PER_EPOCH = 1
	
	
	
config = helmetConfig()
config.display()
# define the model
model = MaskRCNN(mode='training', model_dir='./', config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=1, layers='heads')