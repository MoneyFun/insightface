
from easydict import EasyDict as edict
import os
import json
import numpy as np

def load_property(data_dir):
  prop = edict()
  for line in open(os.path.join(data_dir, 'property')):
    vec = line.strip().split(',')
    assert len(vec)==3
    prop.num_classes = int(vec[0])
    prop.image_size = [int(vec[1]), int(vec[2])]
  return prop
