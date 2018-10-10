# Person-reID-Autoencoder
#### Performing Person re-ID task on images being decompressed with autoencoder. This is the soruce code for my master's thesis, written under supervision of Dr Krystian Mikolajczyk at Imperial College London, Faculty of Electrical and Electronic Engineering.
## Data preparation
I performed all the experiments on CUHK03 dataset, availible at http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html. You can simply download it from the website, unzip and put into a folder which I will refer to as data_path. You may also want to download new protocol for CUHK03, specifically file cuhk03_new_protocol_config_labeled.mat, available at https://github.com/zhunzhong07/person-re-ranking/tree/master/evaluation/data/CUHK03. After these operations your data folder should look like this:  
> data_path
>> cuhk03.mat  
>> README.md  
>> cuhk03_new_protocol_config_labeled.mat
