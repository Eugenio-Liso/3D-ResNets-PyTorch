# 3D ResNets for Action Recognition

# THIS IS A FORKED REPO FROM https://github.com/kenshohara/3D-ResNets-PyTorch

## Requirements

* [PyTorch](http://pytorch.org/)

```bash
conda install pandas=0.25.1 scikit-learn=0.21.2 h5py=2.9.0 pytorch=1.0.0 torchvision=0.2.1 cuda80=1.0 -c soumith
```

* FFmpeg, FFprobe (should be installed by default on a Linux system)

* Python 3

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

## Appunti su comandi di training

- Noi dovremmo dare tutti i dati in training, mentre si fa distinzione fra training-validation-ignorato. Forse questa cosa è da cambiare -> modificare i file per gli splits (es quelli in testTrainMulti_7030_splits) e renderli solo 1-2 oppure modificare `hmdb51_json.py` riga 19 per poter prendere anche quelli con indice 0 (togliere continue)
- Also, per quale motivo dovrei dare un file per volta di annotazione?! `--annotation_path ucf101_01.json` ? Perchè non tutti quanti? Per risolvere questo problema, basterà a target creare solo un file di split (e non 3 come HMDB51)
  - Gli split sono stati decisi in modo tale che differenti video finiscano in run differenti. Si fa poi una media dei risultati
- Bisognerà modificare il parametro `--dataset` nel training per accettare un dataset _generico_.

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

#### TRAINING GENERIC DATASETS

```bash
#genera frames
python util_scripts/generate_video_jpgs.py <dataset_path> <output_path> generic

# TODO
```

### TRAINING 

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

#### HELP

```bash
usage: main.py [-h] [--root_path ROOT_PATH] [--video_path VIDEO_PATH]
               [--annotation_path ANNOTATION_PATH] [--result_path RESULT_PATH]
               [--dataset DATASET] [--n_classes N_CLASSES]
               [--n_pretrain_classes N_PRETRAIN_CLASSES]
               [--pretrain_path PRETRAIN_PATH]
               [--ft_begin_module FT_BEGIN_MODULE] [--sample_size SAMPLE_SIZE]
               [--sample_duration SAMPLE_DURATION]
               [--sample_t_stride SAMPLE_T_STRIDE] [--train_crop TRAIN_CROP]
               [--train_crop_min_scale TRAIN_CROP_MIN_SCALE]
               [--train_crop_min_ratio TRAIN_CROP_MIN_RATIO] [--no_hflip]
               [--colorjitter] [--train_t_crop TRAIN_T_CROP]
               [--learning_rate LEARNING_RATE] [--momentum MOMENTUM]
               [--dampening DAMPENING] [--weight_decay WEIGHT_DECAY]
               [--mean_dataset MEAN_DATASET] [--no_mean_norm] [--no_std_norm]
               [--value_scale VALUE_SCALE] [--nesterov]
               [--optimizer OPTIMIZER] [--lr_scheduler LR_SCHEDULER]
               [--multistep_milestones MULTISTEP_MILESTONES [MULTISTEP_MILESTONES ...]]
               [--overwrite_milestones] [--plateau_patience PLATEAU_PATIENCE]
               [--batch_size BATCH_SIZE]
               [--inference_batch_size INFERENCE_BATCH_SIZE]
               [--n_epochs N_EPOCHS] [--n_val_samples N_VAL_SAMPLES]
               [--resume_path RESUME_PATH] [--no_train] [--no_val]
               [--inference] [--inference_subset INFERENCE_SUBSET]
               [--inference_stride INFERENCE_STRIDE]
               [--inference_crop INFERENCE_CROP] [--inference_no_average]
               [--no_cuda] [--n_threads N_THREADS] [--checkpoint CHECKPOINT]
               [--model MODEL] [--model_depth MODEL_DEPTH]
               [--conv1_t_size CONV1_T_SIZE] [--conv1_t_stride CONV1_T_STRIDE]
               [--no_max_pool] [--resnet_shortcut RESNET_SHORTCUT]
               [--wide_resnet_k WIDE_RESNET_K]
               [--resnext_cardinality RESNEXT_CARDINALITY]
               [--manual_seed MANUAL_SEED] [--accimage]
               [--output_topk OUTPUT_TOPK] [--file_type FILE_TYPE]
               [--tensorboard]

optional arguments:
  -h, --help            show this help message and exit
  --root_path ROOT_PATH
                        Root directory path
  --video_path VIDEO_PATH
                        Directory path of videos
  --annotation_path ANNOTATION_PATH
                        Annotation file path
  --result_path RESULT_PATH
                        Result directory path
  --dataset DATASET     Used dataset (activitynet | kinetics | ucf101 |
                        hmdb51)
  --n_classes N_CLASSES
                        Number of classes (activitynet: 200, kinetics: 400 or
                        600, ucf101: 101, hmdb51: 51)
  --n_pretrain_classes N_PRETRAIN_CLASSES
                        Number of classes of pretraining task.When using
                        --pretrain_path, this must be set.
  --pretrain_path PRETRAIN_PATH
                        Pretrained model path (.pth).
  --ft_begin_module FT_BEGIN_MODULE
                        Module name of beginning of fine-tuning(conv1, layer1,
                        fc, denseblock1, classifier, ...).The default means
                        all layers are fine-tuned.
  --sample_size SAMPLE_SIZE
                        Height and width of inputs
  --sample_duration SAMPLE_DURATION
                        Temporal duration of inputs
  --sample_t_stride SAMPLE_T_STRIDE
                        If larger than 1, input frames are subsampled with the
                        stride.
  --train_crop TRAIN_CROP
                        Spatial cropping method in training. random is
                        uniform. corner is selection from 4 corners and 1
                        center. (random | corner | center)
  --train_crop_min_scale TRAIN_CROP_MIN_SCALE
                        Min scale for random cropping in training
  --train_crop_min_ratio TRAIN_CROP_MIN_RATIO
                        Min aspect ratio for random cropping in training
  --no_hflip            If true holizontal flipping is not performed.
  --colorjitter         If true colorjitter is performed.
  --train_t_crop TRAIN_T_CROP
                        Temporal cropping method in training. random is
                        uniform. (random | center)
  --learning_rate LEARNING_RATE
                        Initial learning rate(divided by 10 while training by
                        lr scheduler)
  --momentum MOMENTUM   Momentum
  --dampening DAMPENING
                        dampening of SGD
  --weight_decay WEIGHT_DECAY
                        Weight Decay
  --mean_dataset MEAN_DATASET
                        dataset for mean values of mean
                        subtraction(activitynet | kinetics)
  --no_mean_norm        If true, inputs are not normalized by mean.
  --no_std_norm         If true, inputs are not normalized by standard
                        deviation.
  --value_scale VALUE_SCALE
                        If 1, range of inputs is [0-1]. If 255, range of
                        inputs is [0-255].
  --nesterov            Nesterov momentum
  --optimizer OPTIMIZER
                        Currently only support SGD
  --lr_scheduler LR_SCHEDULER
                        Type of LR scheduler (multistep | plateau)
  --multistep_milestones MULTISTEP_MILESTONES [MULTISTEP_MILESTONES ...]
                        Milestones of LR scheduler. See documentation of
                        MultistepLR.
  --overwrite_milestones
                        If true, overwriting multistep_milestones when
                        resuming training.
  --plateau_patience PLATEAU_PATIENCE
                        Patience of LR scheduler. See documentation of
                        ReduceLROnPlateau.
  --batch_size BATCH_SIZE
                        Batch Size
  --inference_batch_size INFERENCE_BATCH_SIZE
                        Batch Size for inference. 0 means this is the same as
                        batch_size.
  --n_epochs N_EPOCHS   Number of total epochs to run
  --n_val_samples N_VAL_SAMPLES
                        Number of validation samples for each activity
  --resume_path RESUME_PATH
                        Save data (.pth) of previous training
  --no_train            If true, training is not performed.
  --no_val              If true, validation is not performed.
  --inference           If true, inference is performed.
  --inference_subset INFERENCE_SUBSET
                        Used subset in inference (train | val | test)
  --inference_stride INFERENCE_STRIDE
                        Stride of sliding window in inference.
  --inference_crop INFERENCE_CROP
                        Cropping method in inference. (center | nocrop)When
                        nocrop, fully convolutional inference is performed,and
                        mini-batch consists of clips of one video.
  --inference_no_average
                        If true, outputs for segments in a video are not
                        averaged.
  --no_cuda             If true, cuda is not used.
  --n_threads N_THREADS
                        Number of threads for multi-thread loading
  --checkpoint CHECKPOINT
                        Trained model is saved at every this epochs.
  --model MODEL         (resnet | preresnet | wideresnet | resnext | densenet
                        |
  --model_depth MODEL_DEPTH
                        Depth of resnet (10 | 18 | 34 | 50 | 101)
  --conv1_t_size CONV1_T_SIZE
                        Kernel size in t dim of conv1.
  --conv1_t_stride CONV1_T_STRIDE
                        Stride in t dim of conv1.
  --no_max_pool         If true, the max pooling after conv1 is removed.
  --resnet_shortcut RESNET_SHORTCUT
                        Shortcut type of resnet (A | B)
  --wide_resnet_k WIDE_RESNET_K
                        Wide resnet k
  --resnext_cardinality RESNEXT_CARDINALITY
                        ResNeXt cardinality
  --manual_seed MANUAL_SEED
                        Manually set random seed
  --accimage            If true, accimage is used to load images.
  --output_topk OUTPUT_TOPK
                        Top-k scores are saved in json file.
  --file_type FILE_TYPE
                        (jpg | hdf5)
  --tensorboard         If true, output tensorboard log file.
```