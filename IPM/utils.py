"""
Transforms to Bird-View Plane Utils.
"""

import numpy as np
import math


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[0.0, -direction[2], direction[1]],
                   [direction[2], 0.0, -direction[0]],
                   [-direction[1], direction[0], 0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def get_scaled_homography(H, target_height, estimated_xrange, estimated_yrange):
    # if don't want to scale image, then pass target_height = np.inf

    current_height = estimated_yrange[1] - estimated_yrange[0]
    target_height = min(target_height, current_height)
    (tw, th) = int(np.round((estimated_xrange[1] - estimated_xrange[0]))), int(
        np.round((estimated_yrange[1] - estimated_yrange[0])))

    tr = target_height / float(th)
    target_dim = (int(tw * tr), target_height)

    scaling_matrix = np.array([[tr, 0, 0], [0, tr, 0], [0, 0, 1]])
    scaled_H = np.dot(scaling_matrix, H)

    return scaled_H, target_dim


def get_scaled_matrix(H, target_shape, estimated_xrange, estimated_yrange, strict=False):
    current_height = estimated_yrange[1] - estimated_yrange[0]
    current_width = estimated_xrange[1] - estimated_xrange[0]
    x_scale, y_scael = target_shape[0] / current_width, target_shape[1] / current_height
    if strict:
        scale = min(x_scale, y_scael)
        scaling_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
    else:
        scaling_matrix = np.array([[x_scale, 0, 0], [0, y_scael, 0], [0, 0, 1]])
    scaled_H = np.dot(scaling_matrix, H)
    return scaled_H


def modified_matrices_calculate_range_output_without_translation(height, width, overhead_hmatrix,
                                                                 verbose=False):
    """
    调整透视矩阵对应变换后的图像大小
    :param height:
    :param width:
    :param overhead_hmatrix:
    :param verbose:
    :return:
    """
    range_u = np.array([np.inf, -np.inf])
    range_v = np.array([np.inf, -np.inf])

    i = 0
    j = 0
    u, v, w = np.dot(overhead_hmatrix, [j, i, 1])
    u = u / w
    v = v / w
    out_upperpixel = v
    if verbose:
        print(u, v)
    range_u[0] = min(u, range_u[0])
    range_v[0] = min(v, range_v[0])
    range_u[1] = max(u, range_u[1])
    range_v[1] = max(v, range_v[1])
    i = height - 1
    j = 0
    u, v, w = np.dot(overhead_hmatrix, [j, i, 1])
    u = u / w
    v = v / w
    out_lowerpixel = v
    if verbose:
        print(u, v)
    range_u[0] = min(u, range_u[0])
    range_v[0] = min(v, range_v[0])
    range_u[1] = max(u, range_u[1])
    range_v[1] = max(v, range_v[1])
    i = 0
    j = width - 1
    u, v, w = np.dot(overhead_hmatrix, [j, i, 1])
    u = u / w
    v = v / w
    if verbose:
        print(u, v)
    range_u[0] = min(u, range_u[0])
    range_v[0] = min(v, range_v[0])
    range_u[1] = max(u, range_u[1])
    range_v[1] = max(v, range_v[1])
    i = height - 1
    j = width - 1
    u, v, w = np.dot(overhead_hmatrix, [j, i, 1])
    u = u / w
    v = v / w
    if verbose:
        print(u, v)
    range_u[0] = min(u, range_u[0])
    range_v[0] = min(v, range_v[0])
    range_u[1] = max(u, range_u[1])
    range_v[1] = max(v, range_v[1])

    range_u = np.array(range_u, dtype=np.int)
    range_v = np.array(range_v, dtype=np.int)
    return range_u, range_v


def convertToBirdView(intrinsic, rotation, shape, target_shape, strict=False, verbose=False):
    """
    :return the perspective matrix and target shape of Bird-View Plane
    :param intrinsic:
    :param rotation:
    :param shape:
    :param max_height:
    :param strict:
    :param verbose:
    :return:
    """
    perspective_matrix = intrinsic @ rotation.T @ np.linalg.inv(intrinsic)
    est_range_u, est_range_v = modified_matrices_calculate_range_output_without_translation(shape[1], shape[0],
                                                                                            perspective_matrix,
                                                                                            verbose=verbose)
    moveup_camera = np.array([[1, 0, -est_range_u[0]], [0, 1, -est_range_v[0]], [0, 0, 1]])
    perspective_matrix = np.dot(moveup_camera, perspective_matrix)
    scale_matrix = get_scaled_matrix(perspective_matrix, target_shape, est_range_u, est_range_v, strict=strict)

    return scale_matrix
