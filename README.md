# ReDNetPlus

## Overview

This repo contains the official implementation of ReDNet and ReDNetPlus, a self-attention based semantic segmentation for natural disaster damage assessment.
![alt text](https://github.com/tashchowdhury/ReDNetPlus/blob/main/rednetplus-arch.png?raw=true)


## Dataset

In this repo, we use two datasets: FloodNet and HRUD. The details of these two datasets are presented below.

### FloodNet:

[FloodNet](https://ieeexplore.ieee.org/document/9460988) provides high-resolution UAS imageries with detailed semantic annotation regarding the damages.

The whole dataset has 2343 images, divided into training (~60%), validation (~20%), and test (~20%) sets. The semantic segmentation labels include: 1) Background, 2) Building Flooded, 3) Building Non-Flooded, 4) Road Flooded, 5) Road Non-Flooded, 6) Water, 7)Tree, 8) Vehicle, 9) Pool, 10) Grass. 

The dataset can be downloaded from this link: https://drive.google.com/drive/folders/1leN9eWVQcvWDVYwNb2GCo5ML_wBEycWD?usp=sharing

### HRUD

[HRUD](https://ieeexplore.ieee.org/abstract/document/9377916) provides high-resolution UAS imageries with detailed semantic annotation regarding the damages.

The HRUD dataset is consists of 1973 images. The semantic segmentation labels include: 1) Background, 2) Debris, 3) Water, 4) Building Not Totally Destroyed, 5) Building Totally Destroyed, 6) Vehicle, 7) Road, 8) Tree, 9) Pool, 10) Sand.

The dataset is not publicly released yet.

## Paper Link
If you find the code or trained models useful, please consider citing:

1. The FloodNet paper can be downloaded from this [link](https://ieeexplore.ieee.org/document/9460988).
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

2. The HRUD paper can be downloaded from this [link](https://ieeexplore.ieee.org/abstract/document/9377916).
HRUD paper:

 ```
 @inproceedings{chowdhury2020comprehensive,
  title={Comprehensive semantic segmentation on high resolution uav imagery for natural disaster damage assessment},
  author={Chowdhury, Tashnim and Rahnemoonfar, Maryam and Murphy, Robin and Fernandes, Odair},
  booktitle={2020 IEEE International Conference on Big Data (Big Data)},
  pages={3904--3913},
  year={2020},
  organization={IEEE}
}

```

3. The papers on ReDNet and ReDNetPlus can be downloaded from this [link](https://ieeexplore.ieee.org/abstract/document/9377916).

 ```
@article{chowdhury2021attention,
  title={Attention Based Semantic Segmentation on UAV Dataset for Natural Disaster Damage Assessment},
  author={Chowdhury, Tashnim and Rahnemoonfar, Maryam},
  journal={arXiv preprint arXiv:2105.14540},
  year={2021}
}

@article{chowdhuryattention,
  title={Attention For Damage Assessment},
  author={Chowdhury, Tashnim and Rahnemoonfar, Maryam},
  booktitle={2021 ICML Workshop on Climate Change AI},
  year={2021}
}

```
