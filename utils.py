import numpy as np

# keypoints = {'front_up_right': 0, 'front_up_left': 1, 'front_light_right': 2, 'front_light_left': 3,
#              'front_low_right': 4,
#              'front_low_left': 5, 'central_up_left': 6, 'front_wheel_left': 7, 'rear_wheel_left': 8,
#              'rear_corner_left': 9,
#              'rear_up_left': 10, 'rear_up_right': 11, 'rear_light_left': 12, 'rear_light_right': 13,
#              'rear_low_left': 14,
#              'rear_low_right': 15, 'central_up_right': 16, 'rear_corner_right': 17, 'rear_wheel_right': 18,
#              'front_wheel_right': 19,
#              'rear_plate_left': 20, 'rear_plate_right': 21, 'mirror_edge_left': 22, 'mirror_edge_right': 23}

# horizontal
horizontal_pairs_index = [[0, 1], [2, 3], [4, 5], [6, 16], [7, 19], [8, 18], [9, 17], [10, 11], [12, 13], [14, 15],
                          [20, 21], [22, 23]]

# vertical
vertical_pairs_index = [[0, 11], [1, 10], [2, 13], [3, 12], [4, 15], [5, 14], [7, 8], [18, 19]]


def get_pair_keypoints(index, ktype='horizontal'):
    pair_keypoints = []
    flag = False
    if ktype == 'horizontal':
        pairs = horizontal_pairs_index

    elif ktype == 'vertical':
        pairs = vertical_pairs_index
    else:
        return flag, None
    for count, i in enumerate(index):
        for j in index[count:]:
            if [i, j] in pairs:
                pair_keypoints.append([i, j])
    flag = True if len(pair_keypoints) else False
    return flag, np.array(pair_keypoints)


def get_pair_keypoints_v1(index):
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


def scale_image(shape, dst_shape, scaleUp=False, force=False):
    x_scale = dst_shape[1] / shape[1]
    y_scale = dst_shape[0] / shape[0]
    if not force:
        scale = x_scale if x_scale < y_scale else y_scale
        if not scaleUp:
            scale = min(scale, 1)
        return scale
    else:
        return x_scale, y_scale
