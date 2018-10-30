# from scipy.io import loadmat
# import os.path as osp
# import numpy as np
#
# data_path = 'data/'
# train_idxs = loadmat(osp.join(data_path, "cuhk03_new_protocol_config_labeled.mat"))['train_idx'].flatten() - 1
# img_idxs = loadmat(osp.join(data_path, "cuhk03_new_protocol_config_labeled.mat"))['labels'].flatten()
# filelist = loadmat(osp.join(data_path, "cuhk03_new_protocol_config_labeled.mat"), chars_as_strings=True)['filelist']\
#     .flatten()
#
# for idx in train_idxs:
#     name = filelist[idx]
#     print(name[0])
#     print(type(filelist))
#     if idx==10:
#         exit()

# from os import popen
# rows, columns = popen('stty size', 'r').read().split()
# print(rows, columns)
from time import sleep
from time import time
import sys
from utils.progress_bar import print_progress

start_time = time()
for i in range(1000):
    # sleep(0.01)
    # sys.stdout.write('\r' + str(i))
    # sys.stdout.flush()
    print_progress(i, 1000, suffix='{:.2f}'.format(time() - start_time))