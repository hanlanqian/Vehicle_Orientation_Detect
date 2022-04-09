import numpy as np
import time


def get_pair_keypoints(index):
    pair_keypoints = []
    flag = False
    for i, j in zip(index[:-1], index[1:]):
        if i in pair_keypoints:
            continue
        if i % 2 == 0 and j - i == 1:
            pair_keypoints.append(i)
            pair_keypoints.append(j)
    if pair_keypoints:
        flag = True
        pair_keypoints = np.array(pair_keypoints).reshape(-1, 2)
    return flag, pair_keypoints
