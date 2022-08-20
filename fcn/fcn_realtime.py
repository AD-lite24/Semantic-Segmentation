# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import cv2
import fcn_coco2017_utilities as utils
import onnxruntime
import numpy as np
from PIL import Image

session = onnxruntime.InferenceSession('/Users/adityadandwate/Desktop/Projects/Proper/sem_seg/fcn/fcn-resnet50-12-int8/fcn-resnet50-12-int8.onnx', None)


inputs = session.get_inputs()
outputs = session.get_outputs()

output_names = list(map(lambda output: output.name, outputs))
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(0)
#print('test')

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    #print('test2')
    cv2.imshow("input", frame)
    orig_tensor = np.array(frame)
    #print(type(orig_tensor))
    #print(orig_tensor.shape)
    #print('tensor test')
    img_data = utils.preprocess(orig_tensor)
    #print('preprocess test')
    img_data = img_data.unsqueeze(0)
    img_data = img_data.detach().cpu().numpy()
    #print('test3')

    detections = session.run(output_names, {'input': img_data})
    #print('test4')
    output, aux = detections

    conf, result_img, blended_img, _ = utils.visualize_output(orig_tensor, output[0])
    result_img = np.array(result_img)
    print(type(result_img))

    cv2.imshow("result", result_img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

