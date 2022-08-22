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

## Watershed algorithm
A traditional heuristical approach. Watershed algorithm is based on extracting sure background and foreground and then using markers will make watershed run and detect the exact boundaries.

### Report
Compuation time is fast, but segmentation quality is poor as expected. Edge detection is not too bad, but far too much information is lost. Could work on less complicated images though

## Kmeans clustering
A clustering algorithm which divides the image into color clusters, the number of which can be controlled by adjusting the k value. 

### Report
Segmentation quality is quite decent for an unsupervised algorithm, but tuning the k value requires trial and error though and the same value may not be useful for all scenes. Also since it segments based on color, it can lead to unwanted behaviour when detecting objects with high amount of color variation. It works best with high contrast images with uniformly colour objects. However, its combination with other deep learning and heuristic based approaches could have a lot of potential
