import fcn_coco2017_utilities as utils
from PIL import Image
import numpy as np
import cv2
import onnx
import onnxruntime
from onnx import numpy_helper

print('position tester')

session = onnxruntime.InferenceSession('/Users/adityadandwate/Desktop/Projects/Proper/sem_seg/fcn/fcn-resnet50-12-int8/fcn-resnet50-12-int8.onnx', None)
inputs = session.get_inputs()
outputs = session.get_outputs()

output_names = list(map(lambda output: output.name, outputs))
input_name = session.get_inputs()[0].name

img = Image.open('images/room3.jpg')
img = img.resize((640 ,480), Image.Resampling.LANCZOS)
img.show()
orig_tensor = np.array(img)
print(orig_tensor.shape)
img_data = utils.preprocess(orig_tensor)
img_data = img_data.unsqueeze(0)
img_data = img_data.detach().cpu().numpy()

print(img_data.shape)


detections = session.run(output_names, {'input': img_data})

print("Output shape:", list(map(lambda detection: detection.shape, detections)))

output, aux = detections

conf, result_img, blended_img, _ = utils.visualize_output(orig_tensor, output[0])

print(type(result_img))
result_img.show()
