# 3D ResNets for Action Recognition

# THIS IS A FORKED REPO FROM https://github.com/kenshohara/3D-ResNets-PyTorch

## Requirements

* [PyTorch](http://pytorch.org/)

```bash
#MAKE SURE YOU HAVE CUDA 9.0 AND A NVIDIA DRIVER >= 384.81 
conda install cudatoolkit=9.0 pytorch=1.1.0 future=0.17.1 tensorboard=1.14.0 scikit-image=0.15.0 pandas=0.25.1 scikit-learn=0.21.2 h5py=2.9.0 torchvision=0.2.1 cuda90 -c pytorch

conda install pip
pip install opencv-python==3.4.5.20
```

* FFmpeg, FFprobe (should be installed by default on a Linux system)

* Python 3.7

## Summary

This is the PyTorch code for the following papers:

[
Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh,  
"Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?",  
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 6546-6555, 2018.
](http://openaccess.thecvf.com/content_cvpr_2018/html/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.html)

[
Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh,  
"Learning Spatio-Temporal Features with 3D Residual Networks for Action Recognition",  
Proceedings of the ICCV Workshop on Action, Gesture, and Emotion Recognition, 2017.
](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w44/Hara_Learning_Spatio-Temporal_Features_ICCV_2017_paper.pdf)

This code includes training, fine-tuning and testing on Kinetics, ActivityNet, UCF-101, and HMDB-51.  
**If you want to classify your videos or extract video features of them using our pretrained models,
use [this code](https://github.com/kenshohara/video-classification-3d-cnn-pytorch).**

**The Torch (Lua) version of this code is available [here](https://github.com/kenshohara/3D-ResNets).**  
Note that the Torch version only includes ResNet-18, 34, 50, 101, and 152.

## Pre-trained models

Pre-trained models are available [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M?usp=sharing).  
All models are trained on Kinetics.  
ResNeXt-101 achieved the best performance in our experiments. (See [paper](https://arxiv.org/abs/1711.09577) in details.)

### Performance of the models on Kinetics

This table shows the averaged accuracies over top-1 and top-5 on Kinetics.

| Method | Accuracies |
|:---|:---:|
| ResNet-18 | 66.1 |
| ResNet-34 | 71.0 |
| ResNet-50 | 72.2 |
| ResNet-101 | 73.3 |
| ResNet-152 | 73.7 |
| ResNet-200 | 73.7 |
| ResNet-200 (pre-act) | 73.4 |
| Wide ResNet-50 | 74.7 |
| ResNeXt-101 | 75.4 |
| DenseNet-121 | 70.8 |
| DenseNet-201 | 72.3 |

# Preparation

## Extra datasets support

Follow the instructions in the README under each folder in `datasets`

### ActivityNet

* Download videos using [the official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler).
* Convert from avi to jpg files using ```utils/video_jpg.py```

```bash
python utils/video_jpg.py avi_video_directory jpg_video_directory
```

* Generate fps files using ```utils/fps.py```

```bash
python utils/fps.py avi_video_directory jpg_video_directory
```

### Kinetics

* Download videos using [the official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics).
  * Locate test set in ```video_directory/test```.
* Convert from avi to jpg files using ```utils/video_jpg_kinetics.py```

```bash
python utils/video_jpg_kinetics.py avi_video_directory jpg_video_directory
```

* Generate n_frames files using ```utils/n_frames_kinetics.py```

```bash
python utils/n_frames_kinetics.py jpg_video_directory
```

* Generate annotation file in json format similar to ActivityNet using ```utils/kinetics_json.py```
  * The CSV files (kinetics_{train, val, test}.csv) are included in the crawler.

```bash
python utils/kinetics_json.py train_csv_path val_csv_path test_csv_path dst_json_path
```

### UCF-101

* Download videos and train/test splits [here](http://crcv.ucf.edu/data/UCF101.php).
* Convert from avi to jpg files using ```utils/video_jpg_ucf101_hmdb51.py```

```bash
python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
```

* Generate n_frames files using ```utils/n_frames_ucf101_hmdb51.py```

```bash
python utils/n_frames_ucf101_hmdb51.py jpg_video_directory
```

* Generate annotation file in json format similar to ActivityNet using ```utils/ucf101_json.py```
  * ```annotation_dir_path``` includes classInd.txt, trainlist0{1, 2, 3}.txt, testlist0{1, 2, 3}.txt

```bash
python utils/ucf101_json.py annotation_dir_path
```

### HMDB-51

* Download videos and train/test splits [here](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).
* Convert from avi to jpg files using ```utils/video_jpg_ucf101_hmdb51.py```

```bash
python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
```

* Generate n_frames files using ```utils/n_frames_ucf101_hmdb51.py```

```bash
python utils/n_frames_ucf101_hmdb51.py jpg_video_directory
```

* Generate annotation file in json format similar to ActivityNet using ```utils/hmdb51_json.py```
  * ```annotation_dir_path``` includes brush_hair_test_split1.txt, ...

```bash
python utils/hmdb51_json.py annotation_dir_path
```

## Running the code

Assume the structure of data directories is the following:

```misc
~/
  data/
    kinetics_videos/
      jpg/
        .../ (directories of class names)
          .../ (directories of video names)
            ... (jpg files)
    results/
      save_100.pth
    kinetics.json
```

Confirm all options.

```bash
python main.py -h
```

Train ResNets-34 on the Kinetics dataset (400 classes) with 4 CPU threads (for data loading).  
Batch size is 128.  
Save models at every 5 epochs.
All GPUs is used for the training.
If you want a part of GPUs, use ```CUDA_VISIBLE_DEVICES=...```.

```bash
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --model resnet \
--model_depth 34 --n_classes 400 --batch_size 128 --n_threads 4 --checkpoint 5
```

Continue Training from epoch 101. (~/data/results/save_100.pth is loaded.)

```bash
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --resume_path results/save_100.pth \
--model_depth 34 --n_classes 400 --batch_size 128 --n_threads 4 --checkpoint 5
```

Fine-tuning conv5_x and fc layers of a pretrained model (~/data/models/resnet-34-kinetics.pth) on UCF-101.

```bash
python main.py --root_path ~/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \
--result_path results --dataset ucf101 --n_classes 400 --n_finetune_classes 101 \
--pretrain_path models/resnet-34-kinetics.pth --ft_begin_index 4 \
--model resnet --model_depth 34 --resnet_shortcut A --batch_size 128 --n_threads 4 --checkpoint 5
```

### PREPROCESSING

#### Training su HMDB

- **CURRENT BRANCH (WORK/THESIS)**

```bash
#genera frames
python util_scripts/generate_video_jpgs.py /mnt/external-drive/datasets/hmdb /home/eugenio/Documents/lavoro/git/3D-ResNets-PyTorch/training/hmdb/videos generic

# Crea file di annotazione specifici
#{
#  "labels": [
#    "run"
#  ],
#  "database": {
#    "20060723sfjffangelina_run_f_nm_np1_ri_med_2": {
#      "subset": "training",
#      "annotations": {
#        "label": "run"
#      }
#    }
#  }
#}
python util_scripts/hmdb51_json.py /home/eugenio/Documents/lavoro/git/3D-ResNets-PyTorch/training/hmdb/testTrainMulti_7030_splits_ONLY_RUN_WALK /home/eugenio/Documents/lavoro/git/3D-ResNets-PyTorch/training/hmdb/videos /home/eugenio/Documents/lavoro/git/3D-ResNets-PyTorch/annotations
```



- **OLD MASTER BRANCH**

```bash
python utils_from_master/video_jpg_ucf101_hmdb51.py /mnt/external-drive/datasets/hmdb/ /home/eugenio/Documents/lavoro/git/3D-ResNets-PyTorch/training/hmdb/videos # ffmpeg genera frames per ogni video

python utils_from_master/n_frames_ucf101_hmdb51.py /home/eugenio/Documents/lavoro/git/3D-ResNets-PyTorch/training/hmdb/videos # Genera un file n_frames col numero di frames di ogni video al suo interno

# Al suo interno ci sono NOME_VIDEO <numero>
# 0 -> ignorato | 1 -> training | 2 -> validation nei file contententi gli splits, es. 0_FIRST_DATES_run_f_nm_np1_ba_med_20.avi 1
python utils_from_master/hmdb51_json.py /home/eugenio/Documents/lavoro/git/3D-ResNets-PyTorch/training/hmdb/testTrainMulti_7030_splits # Crea 3 json nella cartella (ad es. hmdb51_1.json)
# Crea un DATABASE di questo tipo:
#{
#  "labels": [
#    "run"
#  ],
#  "database": {
#    "20060723sfjffangelina_run_f_nm_np1_ri_med_2": {
#      "subset": "training",
#      "annotations": {
#        "label": "run"
#      }
#    }
#  }
#}

# SOLO RUN + WALK
python utils_from_master/hmdb51_json.py /home/eugenio/Documents/lavoro/git/3D-ResNets-PyTorch/training_videos/hmdb/testTrainMulti_7030_splits_ONLY_RUN_WALK
###
```

#### PREPROCESSING GENERIC DATASETS

```bash
#genera frames
python util_scripts/generate_video_jpgs.py <dataset_path_subfolders_with_classes> <output_path> generic

python util_scripts/generic_dataset_to_json.py \
--output_annotations_path  
--video_path_training 
--video_path_validation  
--output_splits_path

oppure con --split_size
# TODO
```

### TRAINING 

**N.B**: DO NOT USE A DIFFERENT OPTIMIZER WHEN RESUMING A TRAINING!!!

#### resnet-34

```bash
python main.py  \
--video_path /home/eugenio/Documents/lavoro/git/3D-ResNets-PyTorch/training/hmdb/videos \
--annotation_path /home/eugenio/Documents/lavoro/git/3D-ResNets-PyTorch/training/hmdb/testTrainMulti_7030_splits_ONLY_RUN_WALK/hmdb51_1.json \
--result_path /home/eugenio/Documents/lavoro/git/3D-ResNets-PyTorch/training/results \
--dataset hmdb51 \
--n_classes 2 \
--n_pretrain_classes 400 \
--pretrain_path /home/eugenio/Documents/lavoro/git/3D-ResNets-PyTorch/pretrained/resnet-34-kinetics.pth \
--ft_begin_module fc \
--model resnet \
--model_depth 34 \
--resnet_shortcut A \
--batch_size 64 \
--n_threads 4 \
--checkpoint 5 \
--n_epochs 10
```

#### resnet-101

```bash
python main.py  \
--video_path /home/eugenio/Documents/lavoro/git/3D-ResNets-PyTorch/training/hmdb/videos \
--annotation_path /home/eugenio/Documents/lavoro/git/3D-ResNets-PyTorch/training/hmdb/testTrainMulti_7030_splits_ONLY_RUN_WALK/hmdb51_1.json \ # Ce ne possono essere anche altri (vedere se modificare)
--result_path /home/eugenio/Documents/lavoro/git/3D-ResNets-PyTorch/training/results \
--dataset hmdb51 \
--n_classes 2 \ # Classi in output?
--n_pretrain_classes 400 \ # Classi nel modello pre-trainato (kinetics-400)
--pretrain_path /home/eugenio/Documents/lavoro/git/3D-ResNets-PyTorch/pretrained/resnet-101-kinetics.pth \
--ft_begin_module fc \ # Guarda come sono fatte le classi. Ad esempio, per resnet in resnet.py | self."conv1" etc..
--model resnet \ # N.B: CAMBIARE IN BASE AL MODELLO DELLA RETE
--model_depth 101 \ # N.B: CAMBIARE IN BASE AL MODELLO DELLA RETE
--resnet_shortcut B \ # N.B: CAMBIARE IN BASE AL MODELLO DELLA RETE
--batch_size 64 \ # Default 128. DA TUNARE
--n_threads 4 \
--checkpoint 5 \
--n_epochs 10 # CONF AGGIUNTIVA: DA TUNARE
```