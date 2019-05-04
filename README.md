# Dynamic Sampling

This is the implementation of the paper 'Dynamic Sampling in Convolutional Neural Networks for Imbalanced Data Classification'

## Abstract

Many multimedia systems stream real-time visual data continuously for a wide variety of applications. These systems can produce vast amounts of data, but few studies take advantage of the versatile and real-time data. This paper presents a novel model based on the Convolutional Neural Networks (CNNs) to handle such imbalanced and heterogeneous data and successfully identifies the semantic concepts in these multimedia systems. The proposed model can discover the semantic concepts from the data with a skewed distribution using a dynamic sampling technique. The paper also presents a system that can retrieve real-time visual data from heterogeneous cameras, and the run-time environment allows the analysis programs to process the data from thousands of cameras simultaneously. The evaluation results in comparison with several state-of-the-art methods demonstrate the ability and effectiveness of the proposed model on visual data captured by public network cameras.

## Requirements

- numpy
- keras
- sklearn

## Usage

```
usage: train.py [-h] [--img_size img_height img_width]
                [--valid_batch VALID_BATCH]
                [--batch_per_class BATCH_PER_CLASS]
                [--shear_range SHEAR_RANGE]
                [--horizontal_flip HORIZONTAL_FLIP]
                [--rotation_range ROTATION_RANGE]
                [--width_shift_range WIDTH_SHIFT_RANGE]
                [--height_shift_range HEIGHT_SHIFT_RANGE]
                [--weight_path WEIGHT_PATH] [--epoch EPOCH]
                [--log_path LOG_PATH] [--cflog_path CFLOG_PATH]
                [--cflog_interval CFLOG_INTERVAL]
                [--checkpoint_path CHECKPOINT_PATH] [--warmup]
                [--warmup_epoch WARMUP_EPOCH]
                num_class train_path valid_path

Dynamic Sampling training on Inception-v3-based model

optional arguments:
  -h, --help            show this help message and exit

data:
  num_class             the number of classes in the dataset
  train_path            path to the directory of training images
  valid_path            path to the directory of validation images
  --img_size img_height img_width
                        the target size of input images
  --valid_batch VALID_BATCH
                        batch size during validation
  --batch_per_class BATCH_PER_CLASS
                        batch size per class, batch_size = batch_per_class *
                        num_class

augment:
  --shear_range SHEAR_RANGE
  --horizontal_flip HORIZONTAL_FLIP
  --rotation_range ROTATION_RANGE
  --width_shift_range WIDTH_SHIFT_RANGE
  --height_shift_range HEIGHT_SHIFT_RANGE

model_training:
  --weight_path WEIGHT_PATH
                        path to the model weight file
  --epoch EPOCH         the number of training epoch
  --log_path LOG_PATH   path to the log file of training process
  --cflog_path CFLOG_PATH
                        path to the log file of confusion matrix
  --cflog_interval CFLOG_INTERVAL
                        frequency to log confusion matrix for the whole
                        validation dataset
  --checkpoint_path CHECKPOINT_PATH
                        path to store checkpoint model files

warmup:
  --warmup              set to train the last two layers as warmup process
  --warmup_epoch WARMUP_EPOCH
                        the number of warmup training, valid only when warmup
                        option is used
```

## Reference

[(Pouyanfar et al., 2018)](https://ieeexplore.ieee.org/document/8396983) Pouyanfar, Samira, Yudong Tao, Anup Mohan, Haiman Tian, Ahmed S. Kaseb, Kent Gauen, Ryan Dailey, Sarah Aghajanzadeh, Yung-Hsiang Lu, Shu-Ching Chen, and Mei-Ling Shyu. "Dynamic sampling in convolutional neural networks for imbalanced data classification." In IEEE conference on multimedia information processing and retrieval, pp. 112-117. 2018.

```
@inproceedings{DynamicSampling2018,
  author    = {Samira Pouyanfar and
               Yudong Tao and
               Anup Mohan and
               Haiman Tian and
               Ahmed S. Kaseb and
               Kent Gauen and
               Ryan Dailey and
               Sarah Aghajanzadeh and
               Yung{-}Hsiang Lu and
               Shu{-}Ching Chen and
               Mei{-}Ling Shyu},
  title     = {Dynamic Sampling in Convolutional Neural Networks for Imbalanced Data Classification},
  booktitle = {{IEEE} Conference on Multimedia Information Processing and Retrieval},
  pages     = {112--117},
  year      = {2018}
}
```
