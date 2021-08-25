import os
import cv2
import pickle
import random
import numpy as np
import skimage
import matplotlib.pyplot as plt


"""
import os
import cv2
from random import randint

orj_images_path = "C:/Users/Berk/Desktop/Projeler/Sayac/elektrik_resimleri"
klasor = "test"
images_path = "data/" + klasor + "/input/"
images = os.listdir(images_path)

borderType = cv2.BORDER_CONSTANT

for j in range(2):
    for image in images:
        img = cv2.imread(images_path + image)
        side = [0, 0, 0, 0]
        if randint(0, 1) == 1:
            color = [0, 0, 0]
        else:
            color = [255, 255, 255]
        while sum(side) < 3:
            side = [randint(0, 1), randint(0, 1), randint(0, 1), randint(0, 1)]
        borders = []
        border_height = int(0.4 * img.shape[0])  # shape[0] = rows
        border_width = int(0.4 * img.shape[1])  # shape[1] = cols
        for i, s in enumerate(side):
            if s == 1:
                if i < 2:
                    borders.append(border_height)
                else:
                    borders.append(border_width)
            else:
                borders.append(0)

        dst = cv2.copyMakeBorder(img, borders[0], borders[1], borders[2], borders[3], borderType, None, color)
        cv2.imwrite("data/" + klasor + "/input_border/" + str(j) + "_" + image, dst)

"""

##  Augmentation functions

def rotate(image, rotation_point=None):
    angle = np.random.randint(-5, 6)  # [-5, 6)
    (h, w) = image.shape[:2]
    if rotation_point is None:
        rotation_point = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(rotation_point, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def zoom(image):
    scale = np.random.uniform(0.8, 1.2)  # [0.8, 1.2)
    (h, w) = image.shape[:2]
    rotation_point = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(rotation_point, 0, scale)
    scaled = cv2.warpAffine(image, M, (w, h))
    return scaled

def contrast(image):
    clipLimit = random.randint(1,3)
    rand_grid_size = random.randint(1,8)
    tileGridSize = (rand_grid_size,rand_grid_size)
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2, a, b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return img2

def noisy(image):
    noise_list = ["gauss", "s&p"]
    noise_typ = random.choice(noise_list)
    if noise_typ == "gauss":
        noisy = skimage.util.random_noise(image, mode='gaussian', seed=None, clip=True)
        return noisy
    elif noise_typ == "s&p":
        prob = random.uniform(0, 0.07)
        output = np.zeros(image.shape,np.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output

def brightness(img):
    low, high = 0.5, 3
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1] * value
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2] * value
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def fft2d(image):
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    opt = random.randrange(1, 4, 1)
    coef = 100
    y_value_1, y_value_2 = random.randint(20,50), random.randint(80,110)
    y_value = y_value_1 or y_value_2
    x_value_1, x_value_2 = random.randint(30,110), random.randint(190, 270)
    x_value = x_value_1 or x_value_2

    image_fft = np.fft.fft2(image)
    image = np.fft.fftshift(image_fft)
    if opt == 1: # horizontal
        coef /=10
        image[y_value, center_x] = image[y_value, center_x] * coef
        image[abs(image.shape[0] - y_value), center_x] = image[abs(image.shape[0] - y_value), center_x] * coef

    elif opt == 2: # vertical
        coef /= 4
        image[center_y, x_value] = image[center_y, x_value] * coef
        image[center_y, abs(image.shape[1]-x_value)] = image[center_y, abs(image.shape[1]-x_value)] * coef

    elif opt == 3: # eğik
        image[y_value, x_value] = image[y_value, x_value] * coef
        image[image.shape[0]-y_value, int(image.shape[1]-x_value)] = image[image.shape[0]-y_value, int(
            image.shape[1]-x_value)] * coef

    img_c4 = np.fft.ifftshift(image)
    img_c5 = np.fft.ifft2(img_c4)
    return np.abs(img_c5)

path = r"C:\Users\Golive\Desktop\sayac_project\data\\"
images_path = path + "test\\input\\"
images = os.listdir(images_path)
mask_path = path + "result_small\\"
output_path = r"C:\Users\Golive\Desktop\sayac_project\data\test\\augmented\\"
output_mask_path = r"C:\Users\Golive\Desktop\sayac_project\data\test\\augmented_output\\"
function_list = [rotate, zoom, contrast, brightness, noisy, fft2d]


def main():
    for image in images:
        rand_func = random.choice(function_list)
        print(image, rand_func)
        if rand_func == fft2d:
            orj_img = cv2.imread(images_path + image, 0)
        else:
            orj_img = cv2.imread(images_path + image)
        img = rand_func(orj_img)
        picle = pickle.load(open(mask_path + image.split(".")[0] + ".picle", "rb"))
        if rand_func == rotate or rand_func == zoom:
            all_masks = np.zeros(shape=(128, 320), dtype="uint8")

            for c in range(10):
                mask = picle[:, :, c]
                _, thresh = cv2.threshold(mask, 0.7, 255, cv2.THRESH_BINARY)
                thresh = thresh.astype("uint8")
                all_masks = cv2.bitwise_or(all_masks, thresh)
                mask_aug = rand_func(all_masks)
                pickle.dump(mask_aug, open(output_mask_path + image.split(".")[0] + ".picle", "wb"))
        else:
            pickle.dump(picle, open(output_mask_path + image.split(".")[0] + ".picle", "wb"))
        if rand_func == fft2d:
            plt.imsave(output_path + image, img, cmap="gray")
        else:
            plt.imsave(output_path + image, img)


if __name__ == "__main__":
    main()