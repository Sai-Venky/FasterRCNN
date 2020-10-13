# FasterRCNN

Implements Faster R-CNN Architecture

https://medium.com/@venkysai.96/faster-r-cnn-object-detection-7e48e5b9a906

<ul>
<li><a href="https://github.com/Sai-Venky/FasterRCNN#installation-and-running">Installation and Running</a></li>
<li><a href="https://github.com/Sai-Venky/FasterRCNN#dataset">Dataset</a></li>
<li><a href="https://github.com/Sai-Venky/FasterRCNN#directory-layout">Directory Layout</a></li>
<li><a href="https://github.com/Sai-Venky/FasterRCNN#results">Results</a></li>
<li><a href="https://github.com/Sai-Venky/FasterRCNN#acknowledgement">Acknowledgement</a></li>
<li><a href="https://github.com/Sai-Venky/FasterRCNN#contributing">Contributing</a></li>
<li><a href="https://github.com/Sai-Venky/FasterRCNN#licence">Licence</a></li>
</ul>

### Installation and Running

```pip install requirements.txt```

To train the model, please run 
```python src/train.py```

Once visom is installed, to run it , open a terminal and type ```visdom```
The model can be visualized at ```http://localhost:8097```

### Dataset

The dataset can be downloaded from the following links :-

`wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar`

`wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar`

`wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar`

The dataset can be extracted and stored in the parent directory. If not, its location can be changed in `src/utils/config.py` at `voc_data_dir`

### Directory Layout

The directory structure is as follows :-

* **data :** contains the necessary files needed for loading the VOC dataset along with transformation functions.
  * dataset : base class which instantiates the voc dataset and transforms the raw data.
  * util : helper functions for preprocessing the image, bounding boxes
  * voc_dataset : contains class needed to process/parse the voc dataset data
* **models :** this contains the faster rcnn models and all of its constituent methods.
    * faster_rcnn : core model with train, predict functions. Instantiates and Calls all other models (head, rpn).
    * head : contains the methods needed for VGG head, ROI pooling of faster rcnn.
    * rpn : contains the methods needed for calling region proposal network.
* **utils :** this contains the methods needed for faster rcnn models.
    * anchors : this has all the utility functions related to anchors.
    * config : contains the configuration/options.
    * helper : helper methods
    * proposals : used to generate rpn layer and its corresponding ground truth proposals.
    * visualization : visualization utility functions  

 ### Results

![alt text](https://github.com/Sai-Venky/FasterRCNN/blob/master/imgs/roi_cls_loss.png)
![alt text](https://github.com/Sai-Venky/FasterRCNN/blob/master/imgs/roi_loc_loss.png)
![alt text](https://github.com/Sai-Venky/FasterRCNN/blob/master/imgs/rpn_cls_loss.png)
![alt text](https://github.com/Sai-Venky/FasterRCNN/blob/master/imgs/total_loss.png)
![alt text](https://github.com/Sai-Venky/FasterRCNN/blob/master/imgs/result.png)

 ### Acknowledgement

 https://github.com/chenyuntc/simple-faster-rcnn-pytorch

 ### Contributing

 You can contribute in serveral ways such as creating new features, improving documentation etc.

 ### Licence

 MIT Licence
