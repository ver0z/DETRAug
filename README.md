# DETRAug
Improving small object detection with DETRAug

This repository is the of official implementation of the paper Improving small object detection with DETRAug (**DE**tection **TR**ansformer with **AUG**MIX). Thinking about the way DETR and even Deformable DETR perform their augmentation strategy, we can say that they use a really simple strategy of augmentation. Thinking way ahead we replace part of the original implementation with a more consistent augmentation technique used for classification, known as **AUGMIX**, adapted to object detection we could improve the performance.  

![Fluxo_completo_atualizadi](https://github.com/ver0z/DETRAug/assets/23502680/f15a1717-af6c-4d51-b4bd-b5c921db8697)


## Installation

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n deformable_detr python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate deformable_detr
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```
* To run without issues you must run (with exclamation on Colab)

    ```
    !pip install torch==1.8.0 torchvision==0.9.0 -qq
    ```
### Edit 11/12/2021

* If you are running on a GPU A100-SXM4-40GB that would be better to use the 1.8 version of torch

```
!pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```
* After running the previous commands you must restart your colab

## Usage

### Dataset preparation

We used the same from [Detect waste](https://github.com/wimlds-trojmiasto/detect-waste) as they use the same ways from COCO dataset. All image data is condensed in one file on [Google Drive](https://drive.google.com/file/d/1--3KQlz7-qJVL8UucaqQ10jKiXEBUrix/view?usp=drive_link) 

Please download [COCO 2017 dataset](https://cocodataset.org/) and organize them as following:

```
code_root/
└── data/
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
        	├── instances_train2017.json
        	└── instances_val2017.json
```
* Before start to train adjust the number of classes on deformable_detr.py to your number of classes:
```
  445  #num_classes = 20 if args.dataset_file != 'coco' else 91
  446  #if args.dataset_file == "coco_panoptic":
  447  #    num_classes = 250
    
       num_classes = your_number_of_classes + 1
```

### Training

#### Training on single-GPU

This work were done essentially on a single-GPU the main guidelines will be in this line. In the situation you are using just one GPU you must run the following line, for example (On colab use the exclamation):

```
!python main.py --dataset_file custom --data_path /content/coco --output_dir /content/Output --batch_size 3 --epochs 10
```


#### Some tips to speed-up training
* If your file system is slow to read images, you may consider enabling '--cache_mode' option to load whole dataset into memory at the beginning of training.
* You may increase the batch size to maximize the GPU utilization, according to GPU memory of yours, e.g., set '--batch_size 3' or '--batch_size 4'.



## Citation
It was presented on the International Joint Conference on Neural Networks (IJCNN) 2023 that took place in Queensland, Australia! 
```
@INPROCEEDINGS{10191541,
  author={Cunha, Evair and Macêdo, David and Zanchettin, Cleber},
  booktitle={2023 International Joint Conference on Neural Networks (IJCNN)}, 
  title={Improving small object detection with DETRAug}, 
  year={2023},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/IJCNN54540.2023.10191541}}
```
