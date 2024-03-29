## DFFNet
<span id="jump1"></span>
![CIFReNet Show](./DFFNet.jpg)

###  [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0020025521001389?via%3Dihub)

DFFNet: An IoT-perceptive Dual Feature Fusion Network for General Real-time Semantic Segmentation.<br>

Xiangyan Tang, [Wenxuan Tu](https://github.com/WxTu/CIFReNet/), Keqiu Li, Jieren Cheng.<br>

Information Sciences, 565: 326-343, 2021.<br>

### [License](https://raw.githubusercontent.com/nvlabs/SPADE/master/LICENSE.md)


All rights reserved.
Licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0) 

The code is released for academic research use only. For commercial use, please contact [wenxuantu@163.com].


## Installation

Clone this repo.
```bash
https://github.com/WxTu/DFFNet.git
```

* Windows or Linux
* Python3
* [Pytorch(0.3+)](https://pytorch.org/)
* Numpy
* Torchvision
* Matplotlib


## Preparation

We use [Cityscapes](https://www.cityscapes-dataset.com/), [Camvid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) and [Helen](http://www.f-zhou.com/fa_code.html) datasets. To train a model on these datasets, download datasets from official websites.

Our backbone network is pre-trained on the ImageNet dataset provided by [F. Li et al](http://www.image-net.org/). You can download publically available pre-trained MobileNet v2 from this [website](https://github.com/ansleliu/LightNet).

## Code Structure
- `data/Dataset.py`: processes the dataset before passing to the network.
- `model/DFFNet.py`: defines the architecture of the whole model.
- `model/Backbone.py`: defines the encoder.
- `model/Layers.py`: defines the MFFM, LSPM, and others.
- `utils/Config.py`: defines some hyper-parameters.
- `utils/Process.py`: defines the process of data pretreatment.
- `utils/Utils.py`: defines the loss, optimization, metrics, and others.
- `utils/Visualization.py`: defines the data visualization.
- `Train.py`: the entry point for training and validation.
- `Test.py`: the entry point for testing.

<span id="jump2"></span>

## Visualization
![Visual Show](./visual.jpg)

## Contact
[wenxuantu@163.com](wenxuantu@163.com)

Any discussions or concerns are welcomed!

## Citation
If you use this code for your research, please cite our papers.
```
@article{Tang2021DFFNet,
  title={DFFNet: An IoT-perceptive Dual Feature Fusion Network for General Real-time Semantic Segmentation},
  author={Xiangyan Tang and Wenxuan Tu and Keqiu Li and Jieren Cheng},
  journal={Information Sciences},
  volume={565},
  pages={326-343},
  year={2021}
}
```
## Acknowledgement

[https://github.com/ansleliu/LightNet](https://github.com/ansleliu/LightNet)

[https://github.com/meetshah1995/pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg)

[https://github.com/zijundeng/pytorch-semantic-segmentation](https://github.com/zijundeng/pytorch-semantic-segmentation)

[https://github.com/Tramac/awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)


