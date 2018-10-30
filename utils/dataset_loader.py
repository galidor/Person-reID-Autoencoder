from PIL import Image
from torch.utils.data import Dataset
import h5py
import numpy as np
from scipy.io import loadmat
import os
import math
from scipy.misc import imsave
import os.path as osp


class ImageDataset(Dataset):

    def __init__(self, dataset, transform=None, data_path=False):
        self.dataset = dataset
        self.transform = transform
        self.data_path = data_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, img_idx, camera_id = self.dataset[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.data_path:
            return img, int(img_idx), camera_id, img_path
        return img, int(img_idx), camera_id


class CUHK03(object):
    #Preprocess raw CUHK03 to match PyTorch format
    def __init__(self, data_path, image_path, preprocess=False, preprocess_check=False):
        if preprocess:
            self.preprocess_CUHK03(data_path, image_path)
        if preprocess_check:
            self.preprocess_check_CUHK03(data_path, image_path)
        if osp.exists(image_path):
            datasets = self.get_CUHK03_dataset(data_path, image_path)
            self.train = np.swapaxes(datasets[0], 0, 1)
            self.query = np.swapaxes(datasets[1], 0, 1)
            self.gallery = np.swapaxes(datasets[2], 0, 1)
        else:
            print('Dataset is not preprocessed, use --preprocess_dataset to allow preprocessing, do it only once.')

    def get_CUHK03_dataset(self, data_path, image_path):
        train_idxs = loadmat(osp.join(data_path, "cuhk03_new_protocol_config_labeled.mat"))['train_idx'].flatten()-1
        query_idxs = loadmat(osp.join(data_path, "cuhk03_new_protocol_config_labeled.mat"))['query_idx'].flatten()-1
        gallery_idxs = loadmat(osp.join(data_path, "cuhk03_new_protocol_config_labeled.mat"))['gallery_idx'].flatten()-1
        img_idxs = loadmat(osp.join(data_path, "cuhk03_new_protocol_config_labeled.mat"))['labels'].flatten()
        filelist = loadmat(osp.join(data_path, "cuhk03_new_protocol_config_labeled.mat"))['filelist'].flatten()

        def create_set(image_path, filelist, set_idxs, img_idxs, train=False):
            img_paths = []
            camera_idxs = []
            labels = img_idxs[set_idxs]
            if train:
                train_labels_replaced = range(len(np.unique(img_idxs[set_idxs])))
                train_labels = list(set(img_idxs[set_idxs]))
                for i, n in enumerate(labels):
                    labels[i] = train_labels_replaced[train_labels.index(n)]
            for idx in set_idxs:
                name = filelist[idx][0]
                camera_idxs.append(int(name.split("_")[2]))
                path = osp.join(image_path, name)
                img_paths.append(path)
            return img_paths, labels, camera_idxs

        dataset_train = create_set(image_path, filelist, train_idxs, img_idxs, train=True)
        dataset_query = create_set(image_path, filelist, query_idxs, img_idxs)
        dataset_gallery = create_set(image_path, filelist, gallery_idxs, img_idxs)
        return dataset_train, dataset_query, dataset_gallery

    def preprocess_CUHK03(self, data_path, image_path):
        print("Data preprocessing, saving images to directory {}.".format(image_path))
        try:
            os.makedirs(image_path)
        except OSError:
            if not os.path.isdir(image_path):
                raise
        from progress.bar import IncrementalBar
        f = h5py.File(osp.join(data_path, "cuhk03_release/cuhk-03.mat"), "r")
        imgs_processed = 0
        bar = IncrementalBar('Processing images: ', max=14096, suffix='%(percent).1f%% - %(eta)ds')
        for subset_id, subset_ref in enumerate(f['labeled'][0]):
            subset = f[subset_ref][:].T
            for img_idx in range(subset.shape[0]):
                for view_idx in range(subset.shape[1]):
                    camera_id = 1 if view_idx < 5 else 2
                    img_name = '{:d}_{:03d}_{:d}_{:02d}'.format(subset_id+1, img_idx+1, camera_id, view_idx+1)
                    img = f[subset[img_idx][view_idx]][:].T
                    if img.size == 0 or img.ndim < 3: continue
                    imsave(osp.join(image_path, img_name) + ".png", img)
                    imgs_processed = imgs_processed + 1
                    if imgs_processed%100 == 0:
                        processed = float(imgs_processed)/14096*100
                        remaining = 100 - processed
                        # print("Processing images: [{}{}] {}%".format("#"*int(round(20*processed/100)),
                        #                                              "-"*int(round(20*remaining/100)),
                        #                                              math.ceil(processed)))
                    bar.next()
        bar.finish()
        print("Processing images succeeded.")

    def preprocess_check_CUHK03(self, data_path, image_path):
        print("Checking data path...")
        filelist = loadmat(osp.join(data_path, "cuhk03_new_protocol_config_labeled.mat"))['filelist']
        missing_files = False
        for idx, name in enumerate(filelist.flatten()):
            name = name[0]
            path = osp.join(image_path, name)
            if not os.path.isfile(path):
                print("Missing file: {}".format(path))
                missing_files = True
        if not missing_files: print("Check successful, no missing files.")