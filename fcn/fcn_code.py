import fcn_coco2017_utilities as utils

import json
import sys
import os
import time
import numpy as np
import cv2
import onnx
import onnxruntime
from onnx import numpy_helper
from matplotlib.pyplot import imshow

session = onnxruntime.InferenceSession('/Users/adityadandwate/Desktop/Projects/Proper/sem_seg/fcn/fcn-resnet50-12-int8/fcn-resnet50-12-int8.onnx', None)
inputs = session.get_inputs()
outputs = session.get_outputs()

output_names = list(map(lambda output: output.name, outputs))
input_name = session.get_inputs()[0].name


detections = session.run(output_names, {'input': utils.img_data})

print("Output shape:", list(map(lambda detection: detection.shape, detections)))

output, aux = detections

conf, result_img, blended_img, _ = utils.visualize_output(utils.orig_tensor, output[0])

print(type(result_img))
result_img.show()

#conf, result_img, blended_img, raw_labels = utils.visualize_output(utils.img_data, result[0])