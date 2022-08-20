# Semantic Segmentation

Following models were tested 

## Enet

Tested on the Cityscapes dataset, both on still pictures and real time camera feed. 

### Report
Very lightweight, so great real time performance. However, segmentation quality is passable at best. Model can be enhanced if trained on a less complicated, custom dataset. 

The paper can be found here https://arxiv.org/abs/1606.02147

## FCN

Tested on the COCO 2017 dataset, on still images. The model tested was quantized.

### Report
Model is heavy, but the segmentation quality is great. Works best when image resolution is around 640x480, equal to the resolution of the coco dataset. Code to fit resolution has been added. Real time performance is shaky but passable, but quality is far better than the likes of enet. 

The paper can be found here https://arxiv.org/abs/1411.4038
