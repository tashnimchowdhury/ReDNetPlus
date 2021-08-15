# ReDNetPlus

## Overview

This repo contains the official implementation of ReDNet and ReDNetPlus, a self-attention based semantic segmentation for natural disaster damage assessment.

![alt text](https://github.com/tashchowdhury/ReDNetPlus/files/6988958/rednetplus-arch.pdf)


## Dataset Details

Two datasets are implemented. FloodNet and RescueNet.

### FloodNet:

[FloodNet](https://ieeexplore.ieee.org/document/9460988) provides high-resolution UAS imageries with detailed semantic annotation regarding the damages.

The data is collected with a small UAS platform, DJI Mavic Pro quadcopters, after Hurricane Harvey. The whole dataset has 2343 images, divided into training (~60%), validation (~20%), and test (~20%) sets. The semantic segmentation labels include: 1) Background, 2) Building Flooded, 3) Building Non-Flooded, 4) Road Flooded, 5) Road Non-Flooded, 6) Water, 7)Tree, 8) Vehicle, 9) Pool, 10) Grass. 

The dataset can be downloaded from this link: https://drive.google.com/drive/folders/1leN9eWVQcvWDVYwNb2GCo5ML_wBEycWD?usp=sharing

### RescueNet

## Paper Link
The FloodNet paper can be downloaded from this [link](https://ieeexplore.ieee.org/document/9460988).
FloodNet paper:

 ```
 @ARTICLE{9460988,
  author={Rahnemoonfar, Maryam and Chowdhury, Tashnim and Sarkar, Argho and Varshney, Debvrat and Yari, Masoud and Murphy, Robin Roberson},
  journal={IEEE Access}, 
  title={FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding}, 
  year={2021},
  volume={9},
  number={},
  pages={89644-89654},
  doi={10.1109/ACCESS.2021.3090981}
  }
 
@article{rahnemoonfar2020floodnet,
  title={FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding},
  author={Rahnemoonfar, Maryam and Chowdhury, Tashnim and Sarkar, Argho and Varshney, Debvrat and Yari, Masoud and Murphy, Robin},
  journal={arXiv preprint arXiv:2012.02951},
  year={2020}
}

```
