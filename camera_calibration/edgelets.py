"""
检测车辆横向边缘
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from camera_calibration.diamondSpace import DiamondSpace


def neighborhood(grad, winSize=9):
    l = []
    if winSize % 2 != 1:
        print("winSize should be an odd number.")
        raise TypeError
    low = int((winSize - 1) / 2)
    top = int((winSize + 1) / 2)
    for i in range(-low, top):
        l.append(np.stack([np.arange(-low, top).reshape(1, -1), np.full((1, winSize), i)], axis=2))
    weight = np.vstack(l)
    grad_pad = np.pad(grad, ((low, low), (low, low)), 'constant')
    height, width = grad_pad.shape[:2]
    orientations = np.zeros((height - 2 * low, width - 2 * low, 2))
    quality = np.zeros_like(grad)

    # y_index, x_index = np.where(grad_pad != 0)
    # start = time()
    # for y in y_index:
    #     for x in x_index:
    for y in range(height):
        for x in range(width):
            if grad_pad[y][x] == 0:
                continue
            window = grad_pad[y - low:y + top, x - low:x + top]
            x_weight = np.multiply(window, weight[:, :, 0]).ravel()
            y_weight = np.multiply(window, weight[:, :, 1]).ravel()

            o, q = calculate(np.stack([x_weight, y_weight], axis=1))
            orientations[y - low][x - low] = o
            quality[y - low][x - low] = q
    # print(f'time is {time()-start}')
    return orientations, quality


def accumulate_orientation(orientations, quality, threshold=0.25):
    mask = np.zeros_like(quality)
    thres = np.percentile(quality[quality != 0], 100 * (1 - threshold))
    y_index, x_index = np.where(quality > thres)
    height, width = mask.shape[:2]
    for i, j in zip(y_index, x_index):
        dx, dy = orientations[i, j]
        if dx < 1e-6:
            continue
        slope = dy / dx
        x = np.arange(j - 4, j + 4)
        y = np.around(i - slope * j + slope * x).astype(np.int)
        if (x < width).all() and (y < height).all():
            try:
                mask[y, x] += quality[i, j]
            except Exception as e:
                print(e)
    return mask, thres


def calculate(X, ):
    u, s, v = np.linalg.svd(X.T @ X)
    W = v
    sigma = s.transpose() * s
    orientation = W[:, 0]
    quality = sigma[0] / sigma[1] if sigma[1] != 0 else 0
    return orientation, quality


def get_lines(edges, orientation, box=None):
    y, x = np.where(edges > 0)
    lines = []
    if box is not None:
        real_x = x + box[0]
        real_y = y + box[1]
        for i, j in zip(x, y):
            dx, dy = orientation[j, i]
            a = dy
            b = -dx
            c = (j + box[1]) * dx - dy * (i + box[0])
            # k = IPM[j, i][1] / IPM[j, i][0]
            # b = read_y - k * real_x
            lines.append([a, b, c])
    else:
        for i, j in zip(x, y):
            k = orientation[j, i][1] / orientation[j, i][0]
            b = j - k * i
            lines.append([k, -1, b])
            # start = np.round(index + 5 * IPM[j, i]).astype(np.int32)
            # end = np.round(index - 5 * IPM[j, i]).astype(np.int32)
            # cv2.line(img, start, end, (0, 0, 255), 1)
    return np.vstack(lines)


if __name__ == '__main__':
    DS = DiamondSpace(720)

    # path = './camera_calibration/car.jpg'
    path = 'pics/cars.jpg'
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 700, 700, L2gradient=True)
    orientation, quality = neighborhood(edges)
    thres, edges = cv2.threshold(quality, 0, 255, cv2.THRESH_OTSU)
    lines = get_lines(edges, orientation)
    peak = (147.45110199, -238.7251731)
    DS.filter_lines_from_peak(peak, lines)
    p, v, p_ds = DS.find_peaks(t=0.9)

    A = DS.attach_spaces()
    _, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(A, cmap="Greys", extent=(
        (-DS.size + 0.5) / DS.scale, (DS.size - 0.5) / DS.scale, (DS.size - 0.5) / DS.scale,
        (-DS.size + 0.5) / DS.scale))
    ax[0].set(title="Accumulator", xticks=np.linspace(-DS.size + 1, DS.size - 1, 5) / DS.scale,
              yticks=np.linspace(-DS.size + 1, DS.size - 1, 5) / DS.scale)
    ax[0].plot(p_ds[:, 0] / DS.scale, p_ds[:, 1] / DS.scale, "r+")
    plt.figure()
    print(img.shape)
    print(p)
    ax[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # ax.plot(peak[0], peak[1], 'r+')
    ax[1].plot(p[0, 0], p[0, 1], 'r+')
    ax[0].invert_yaxis()
    plt.show()
