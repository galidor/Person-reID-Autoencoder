# Person-reID-Autoencoder
#### Performing Person re-ID task on images being decompressed with autoencoder. This is the soruce code for my master's thesis, written under supervision of Dr Krystian Mikolajczyk at Imperial College London, Faculty of Electrical and Electronic Engineering.
## Table of Contents:
1. [Data preparation](https://github.com/galidor/Person-reID-Autoencoder#data-preparation)  
2. [Training re-ID feature extractor](https://github.com/galidor/Person-reID-Autoencoder#training-re-id-feature-extractor)  
## Data preparation
I performed all the experiments on CUHK03 dataset, availible [here](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html). You can simply download it from the website, unzip and put into a folder which I will refer to as data_path. You may also want to download new protocol for CUHK03, specifically file cuhk03_new_protocol_config_labeled.mat, available [here](https://github.com/zhunzhong07/person-re-ranking/tree/master/evaluation/data/CUHK03). After these operations your data folder should look like this:  
> data_path
>> cuhk03_new_protocol_config_labeled.mat  
>> cuhk03_release  
>>> cuhk03.mat  
>>> README.md  

After prepairing the folder in such a manner, run the train_model_ResNet50.py program as follows:  
`python train_model_ResNet50.py --data_path /your/data/path --preprocess_dataset`  
Your data_path should then look like this:  
> data_path
>> cuhk03_new_protocol_config_labeled.mat  
>> img  
>> cuhk03_release  
>>> cuhk03.mat  
>>> README.md  

All the images from CUHK03 "labeled" are stored in the folder img.  
## Training re-ID feature extractor
After finishing that you are ready for the next step which is training person re-ID feature extractor by running either train_model_ResNet50.py or train_model_PCB.py as follows (please note that certain arguments may be optional, just type `python train_model_ResNet50.py --help` for detailed description):  
```
python train_model_ResNet50.py --data_path /your/data/path --batch_size 16 --model_path your/model/path --optim_step 20 --learining_rate 0.01 --epochs 50 --normalize --reranking  
python train_model_PCB.py --data_path /your/data/path --batch_size 16 --model_path your/model/path --optim_step 20 --learining_rate 0.01 --epochs 50 --normalize --reranking  
```
