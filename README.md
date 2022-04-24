<div align="center">
<p>
   <a align="left" href="https://ultralytics.com/yolov5" target="_blank">
   <img width="850" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/splash.jpg"></a>
</p>
<br>
<div>
   <a href="https://github.com/ultralytics/yolov5/actions"><img src="https://github.com/ultralytics/yolov5/workflows/CI%20CPU%20testing/badge.svg" alt="CI CPU testing"></a>
   <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv5 Citation"></a>
   <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>
   <br>
   <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
   <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
   <a href="https://join.slack.com/t/ultralytics/shared_invite/zt-w29ei8bp-jczz7QYUmDtgo6r6KcMIAg"><img src="https://img.shields.io/badge/Slack-Join_Forum-blue.svg?logo=slack" alt="Join Forum"></a>
</div>

<br>
<p>
YOLOv5 🚀 is a family of object detection architectures and models pretrained on the COCO dataset, and represents <a href="https://ultralytics.com">Ultralytics</a>
 open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.
</p>

<div align="center">
   <a href="https://github.com/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-github.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://www.linkedin.com/company/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-linkedin.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://twitter.com/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-twitter.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://www.producthunt.com/@glenn_jocher">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-producthunt.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://youtube.com/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-youtube.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://www.facebook.com/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-facebook.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://www.instagram.com/ultralytics/">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-instagram.png" width="2%"/>
   </a>
</div>

<!--
<a align="center" href="https://ultralytics.com/yolov5" target="_blank">
<img width="800" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-api.png"></a>
-->

</div>

## <div align="center">Documentation</div>

See the [YOLOv5 Docs](https://docs.ultralytics.com) for full documentation on training, testing and deployment.

## <div align="center">Quick Start Examples</div>

This tutorial is based on the YOLOv5 repository by Ultralytics. This repository shows training on your own custom objects.

To train custom dataset using YOLOv5 take the following steps:

* Install YOLOv5 dependencies
* Download custom YOLOv5 dataset
* Prepare Pre-Trained Weights
* Custom dataset YOLOv5 training
* Evaluate YOLOv5 performance
* Visualize YOLOv5 training data
* Run YOLOv5 inference on test images
* Export saved YOLOv5 weights

## Training Preparations

Install YOLOv5 dependencies
<details><summary> <b>Expand</b> </summary>

* clone YOLOv5 repository
      
      !git clone https://github.com/ultralytics/yolov5  # clone
      %cd yolov5
  
* Install necessary dependencies
      
      !pip install -qr requirements.txt

</details>

Download custom YOLOv5 dataset
<details><summary> <b>Expand</b> </summary>
  
* Install Roboflow dependencies
  
      !pip install -q roboflow
      from roboflow import Roboflow
      rf = Roboflow(model_format="yolov5", notebook="roboflow-yolor")
  
* Download Dataset from Roboflow
    
      %cd /content/yolov5
      from roboflow import Roboflow
      rf = Roboflow(api_key="85cNlMEKyhhCdduuKla4")
      project = rf.workspace("joseph-nelson").project("uno-cards")
      dataset = project.version(3).download("yolov5")
  
* See YAML category/class
      
      %cat {dataset.location}/data.yaml
   
* Set-up Environment
   
      import os
      os.environ["DATASET_DIRECTORY"] = "/content/datasets"

</details>

Prepare Pre-trained weight
<details><summary> <b>Expand</b> </summary>

* Get pretrained YOLOv5s.pt
      
      import torch
      # Model
      model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

</details>

## Training

Here, we are able to pass a number of arguments:

* img: define input image size
* batch: determine batch size
* epochs: define the number of training epochs. (Note: often, 3000+ are common here!)
* data: Our dataset locaiton is saved in the dataset.location
* weights: specify a path to weights to start transfer learning from. Here we choose the generic COCO pretrained checkpoint.
* cache: cache images for faster training

```
%cd /content/yolov5
!python train.py --img 416 --batch 16 --epochs 150 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache
```

## Evaluate Custom YOLOv5 Detector Performance

Evaluate custom YOLOv5 model
<details><summary> <b>Expand</b> </summary>
 
* Start Tensorboard, run after training is finished. Logs save in runs folder
      
      %load_ext tensorboard
      %tensorboard --logdir runs
  
* Plots data, if tensorboard isn't working
  
      from IPython.display import Image
      from utils.plots import plot_results  # plot results.txt as results.png
      Image(filename='/content/yolov5/runs/train/yolov5s/results.png', width=1000)  # view results.png
  
* Display ground data
      
      print("GROUND TRUTH TRAINING DATA:")
      Image(filename='/content/yolov5/runs/train/yolov5s/train_batch0.jpg', width=900)
      
* Display augmented data
      
      print("AUGMENTED DATA:")
      Image(filename='/content/yolov5/runs/train/yolov5s/train_batch0.jpg', width=900)
      
</details>

Run Inference with trained weight
<details><summary> <b>Expand</b> </summary>
      
* See directory in runs trained folder
    
      %ls runs/train/yolov5s/weights
      
* Create names file for model
    
      import yaml
      import ast
      with open("/content/yolov5/Uno-Cards-3/data.yaml", 'r') as stream:
          names = str(yaml.safe_load(stream)['names'])

      namesFile = open("../data.names", "w+")
      names = ast.literal_eval(names)
      for name in names:
        namesFile.write(name +'\n')
      namesFile.close()
      
* Runs Trained Model with Test Images
      
      !python detect.py --weights runs/train/exp/weights/best.pt --img 416 --conf 0.1 --source {dataset.location}/test/images
     
* Display inference on All Test Images
      
      import glob
      from IPython.display import Image, display

      for imageName in glob.glob('/content/yolov5/inference/output/*.jpg'): #assuming JPG
          display(Image(filename=imageName))
          print("\n")
      
</details>

## Export Trained Weight to GDrive
* Login to GDrive
    
      from google.colab import drive
      drive.mount('/content/gdrive')
  
* Export Trained Weight
  
      %cp /content/yolov5/runs/train/yolov5s/weights/best.pt /content/gdrive/My Drive
    
## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [Ultralytics YOLOv5 Repository](https://github.com/ultralytics/yolov5)

</details>


<details open>
<summary>Tutorials</summary>

* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp; 🚀 RECOMMENDED
* [Tips for Best Training Results](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)&nbsp; ☘️
  RECOMMENDED
* [Weights & Biases Logging](https://github.com/ultralytics/yolov5/issues/1289)&nbsp; 🌟 NEW
* [Roboflow for Datasets, Labeling, and Active Learning](https://github.com/ultralytics/yolov5/issues/4975)&nbsp; 🌟 NEW
* [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
* [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)&nbsp; ⭐ NEW
* [TFLite, ONNX, CoreML, TensorRT Export](https://github.com/ultralytics/yolov5/issues/251) 🚀
* [Test-Time Augmentation (TTA)](https://github.com/ultralytics/yolov5/issues/303)
* [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
* [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
* [Hyperparameter Evolution](https://github.com/ultralytics/yolov5/issues/607)
* [Transfer Learning with Frozen Layers](https://github.com/ultralytics/yolov5/issues/1314)&nbsp; ⭐ NEW
* [Architecture Summary](https://github.com/ultralytics/yolov5/issues/6998)&nbsp; ⭐ NEW

</details>

## <div align="center">Environments</div>

Get started in seconds with our verified environments. Click each icon below for details.

<div align="center">
    <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-colab-small.png" width="15%"/>
    </a>
    <a href="https://www.kaggle.com/ultralytics/yolov5">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-kaggle-small.png" width="15%"/>
    </a>
    <a href="https://hub.docker.com/r/ultralytics/yolov5">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-docker-small.png" width="15%"/>
    </a>
    <a href="https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-aws-small.png" width="15%"/>
    </a>
    <a href="https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-gcp-small.png" width="15%"/>
    </a>
</div>

## <div align="center">Integrations</div>

<div align="center">
    <a href="https://wandb.ai/site?utm_campaign=repo_yolo_readme">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-wb-long.png" width="49%"/>
    </a>
    <a href="https://roboflow.com/?ref=ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-roboflow-long.png" width="49%"/>
    </a>
</div>

|Weights and Biases|Roboflow ⭐ NEW|
|:-:|:-:|
|Automatically track and visualize all your YOLOv5 training runs in the cloud with [Weights & Biases](https://wandb.ai/site?utm_campaign=repo_yolo_readme)|Label and export your custom datasets directly to YOLOv5 for training with [Roboflow](https://roboflow.com/?ref=ultralytics) |


<!-- ## <div align="center">Compete and Win</div>

We are super excited about our first-ever Ultralytics YOLOv5 🚀 EXPORT Competition with **$10,000** in cash prizes!

<p align="center">
  <a href="https://github.com/ultralytics/yolov5/discussions/3213">
  <img width="850" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-export-competition.png"></a>
</p> -->

## <div align="center">Why YOLOv5</div>

<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/155040763-93c22a27-347c-4e3c-847a-8094621d3f4e.png"></p>
<details>
  <summary>YOLOv5-P5 640 Figure (click to expand)</summary>

<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/155040757-ce0934a3-06a6-43dc-a979-2edbbd69ea0e.png"></p>
</details>
<details>
  <summary>Figure Notes (click to expand)</summary>

* **COCO AP val** denotes mAP@0.5:0.95 metric measured on the 5000-image [COCO val2017](http://cocodataset.org) dataset over various inference sizes from 256 to 1536.
* **GPU Speed** measures average inference time per image on [COCO val2017](http://cocodataset.org) dataset using a [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) V100 instance at batch-size 32.
* **EfficientDet** data from [google/automl](https://github.com/google/automl) at batch size 8.
* **Reproduce** by `python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n6.pt yolov5s6.pt yolov5m6.pt yolov5l6.pt yolov5x6.pt`
</details>

### Pretrained Checkpoints

[assets]: https://github.com/ultralytics/yolov5/releases

[TTA]: https://github.com/ultralytics/yolov5/issues/303

|Model |size<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>val<br>0.5 |Speed<br><sup>CPU b1<br>(ms) |Speed<br><sup>V100 b1<br>(ms) |Speed<br><sup>V100 b32<br>(ms) |params<br><sup>(M) |FLOPs<br><sup>@640 (B)
|---                    |---  |---    |---    |---    |---    |---    |---    |---
|[YOLOv5n][assets]      |640  |28.0   |45.7   |**45** |**6.3**|**0.6**|**1.9**|**4.5**
|[YOLOv5s][assets]      |640  |37.4   |56.8   |98     |6.4    |0.9    |7.2    |16.5
|[YOLOv5m][assets]      |640  |45.4   |64.1   |224    |8.2    |1.7    |21.2   |49.0
|[YOLOv5l][assets]      |640  |49.0   |67.3   |430    |10.1   |2.7    |46.5   |109.1
|[YOLOv5x][assets]      |640  |50.7   |68.9   |766    |12.1   |4.8    |86.7   |205.7
|                       |     |       |       |       |       |       |       |
|[YOLOv5n6][assets]     |1280 |36.0   |54.4   |153    |8.1    |2.1    |3.2    |4.6
|[YOLOv5s6][assets]     |1280 |44.8   |63.7   |385    |8.2    |3.6    |12.6   |16.8
|[YOLOv5m6][assets]     |1280 |51.3   |69.3   |887    |11.1   |6.8    |35.7   |50.0
|[YOLOv5l6][assets]     |1280 |53.7   |71.3   |1784   |15.8   |10.5   |76.8   |111.4
|[YOLOv5x6][assets]<br>+ [TTA][TTA]|1280<br>1536 |55.0<br>**55.8** |72.7<br>**72.7** |3136<br>- |26.2<br>- |19.4<br>- |140.7<br>- |209.8<br>-

<details>
  <summary>Table Notes (click to expand)</summary>

* All checkpoints are trained to 300 epochs with default settings. Nano and Small models use [hyp.scratch-low.yaml](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-low.yaml) hyps, all others use [hyp.scratch-high.yaml](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-high.yaml).
* **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](http://cocodataset.org) dataset.<br>Reproduce by `python val.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`
* **Speed** averaged over COCO val images using a [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) instance. NMS times (~1 ms/img) not included.<br>Reproduce by `python val.py --data coco.yaml --img 640 --task speed --batch 1`
* **TTA** [Test Time Augmentation](https://github.com/ultralytics/yolov5/issues/303) includes reflection and scale augmentations.<br>Reproduce by `python val.py --data coco.yaml --img 1536 --iou 0.7 --augment`

</details>
