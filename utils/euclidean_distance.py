import numpy as np
from scipy.spatial.distance import cdist


def calculate_distance(probFea, galFea):
    query_num = probFea.shape[0]
    print(query_num)
    all_num = query_num + galFea.shape[0]
    feat = np.append(probFea,galFea,axis = 0)
    feat = feat.astype(np.float16)
    print('computing original distance')
    original_dist = cdist(feat,feat).astype(np.float16)
    original_dist = np.power(original_dist,2).astype(np.float16)
    del feat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist/np.max(original_dist,axis = 0))
    final_dist = original_dist[:query_num, query_num:]
    return final_dist