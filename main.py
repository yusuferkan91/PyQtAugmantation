import sys
import os
#from os.path import expanduser
import pickle
import random
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import skimage
#import matplotlib.pyplot as plt
from Gui import Ui_MainWindow
#from preview import Ui_Preview

# class Mainpreview(QWidget, Ui_Preview):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.setupUi(self)
#     def imageLoad(self,imgOrj,imgAug,info):
#         self.img_orjinal.setPixmap(imgOrj)
#         self.img_aug.setPixmap(imgAug)
#
#     def closeEvent(self, *args, **kwargs):
#         window.show()


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.btn_start.setEnabled(False)
        self.progressBar.setVisible(False)

        """Variables"""

        self.images_path = ""
        self.mask_path = ""
        self.output_path = ""
        self.output_mask_path = ""
        self.function = []
        self.listSettings = []
        self.expected_number = 0

        """Button clicks"""

        self.btn_img_path.clicked.connect(self.btn_img_path_click)
        self.btn_imgOut_path.clicked.connect(self.btn_imgOut_path_click)
        self.btn_maskOut_path.clicked.connect(self.btn_maskOut_path_click)
        self.btn_mask_path.clicked.connect(self.btn_mask_path_click)
        self.btn_save.clicked.connect(self.btn_save_click)
        self.btn_preview.clicked.connect(self.btn_preview_click)
        self.btn_start.clicked.connect(self.btn_start_click)

        """Checkboxes"""

        self.check_Rotate.clicked.connect(self.Rotate_check)
        self.check_Contrast.clicked.connect(self.Contrast_check)
        self.check_Brightess.clicked.connect(self.Brightess_check)
        self.check_fft2d.clicked.connect(self.fft2d_check)
        self.check_Noisy.clicked.connect(self.Noisy_check)
        self.check_Zoom.clicked.connect(self.Zoom_check)
        self.check_mask_images.clicked.connect(self.mask_images_check)
        self.check_pad.clicked.connect(self.pad_check)

    def btn_start_click(self):
        try:
            images = os.listdir(self.images_path)
            self.progressBar.setEnabled(True)
            count = 0
            while count < self.expected_number:
                np.random.shuffle(images)
                for image in images:
                    self.progress_bar(count, self.expected_number)
                    fs = ""
                    func_to_exec = [np.random.randint(0, 2) for _ in range(len(self.function))]
                    while sum(func_to_exec) < len(self.function)-3:  # 7 func seçilirse en az 4ü çalışır.
                        func_to_exec = [np.random.randint(0, 2) for _ in range(len(self.function))]
                    orj_img = cv2.imread(self.images_path + image)
                    cpy_img = orj_img.copy()
                    if self.check_mask_images.isChecked():
                        orj_mask = pickle.load(open(self.mask_path + image.split(".")[0] + ".picle", "rb"))
                        cpy_mask = orj_mask.copy()
                    else:
                        cpy_mask = None
                    for i in range(len(self.function)):
                        if func_to_exec[i] == 1:
                            fs += self.function[i].__name__[0]
                            cpy_img, cpy_mask = self.function[i](cpy_img, cpy_mask)

                    cv2.imwrite(self.output_path + image.split(".")[0] + "_" + fs + "_" + str(count) + ".jpg",
                                cpy_img.astype("uint8"))
                    if self.check_mask_images.isChecked():
                        pickle.dump(cpy_mask,
                                    open(self.output_mask_path + image.split(".")[0] + "_" + fs + "_" + str(count) + ".picle", "wb"))
                    count += 1
                    if count >= self.expected_number:
                        break
            self.listView.addItem("Augmentation Done")
            self.btn_start.setEnabled(False)
            self.frame.setEnabled(True)
            self.check_mask_images.setEnabled(True)
            self.progressBar.setVisible(False)
        except Exception as e:
            print(e)

    def btn_preview_click(self):
        msg = QMessageBox()
        msg.setWindowTitle("Preview")
        txtMsg = "Applied:\n"
        for each in self.listSettings:
            if each == "ROTATE":
                each = each + "{ MIN: " + str(self.spin_Rotate_min.value()) + " MAX: " + str(
                    self.spin_Rotate_max.value()) + "}\n"
            if each == "PAD":
                each = each + "{ MIN: " + str(self.spin_pad_min.value()) + " MAX: " + str(
                    self.spin_pad_max.value()) + "}\n"
            if each == "CONTRASST":
                each = each + "{ MIN: " + str(self.spin_Contrast_min.value()) + " MAX: " + str(
                    self.spin_Contrast_max.value()) + "}\n"
            if each == "BRİGHTESS":
                each = each + "{ MIN: " + str(self.spin_Brightness_min.value()) + " MAX: " + str(
                    self.spin_Brightness_max.value()) + "}\n"
            if each == "ZOOM":
                each = each + "{ MIN: " + str(self.spin_Zoom_min.value()) + " MAX: " + str(
                    self.spin_Zoom_max.value()) + "}\n"
            txtMsg = txtMsg + each
        msg.setDetailedText(txtMsg)
        msg.setStandardButtons(QMessageBox.Retry | QMessageBox.Close)
        try:
            if len(self.listSettings) <= 0:
                self.listView.addItem("Augmentation NULL")
            else:
                # image_lena = QPixmap("lena.jpg")
                # pre.img_orjinal.setPixmap(image_lena)
                # pre.show()
                # window.hide()
                # images = os.listdir(self.images_path)
                # image = np.random.choice(images)
                # func_to_exec = [np.random.randint(0, 2) for _ in range(len(self.function))]
                # while sum(func_to_exec) < len(self.function) - 3:  # 7 func seçilirse en az 4ü çalışır.
                #     func_to_exec = [np.random.randint(0, 2) for _ in range(len(self.function))]
                # orj_img = cv2.imread(self.images_path + image)
                # cpy_img = orj_img.copy()
                # cpy_mask = None
                # for i in range(len(self.function)):
                #     if func_to_exec[i] == 1:
                #         cpy_img, cpy_mask = self.function[i](cpy_img, cpy_mask)
                # w = 450
                # h = 450
                # orj_img = cv2.resize(orj_img, (w, h))
                # cpy_img = cv2.resize(cpy_img, (w, h))
                # img_or = self.img_to_pixmap(orj_img)
                # img_aug = self.img_to_pixmap(cpy_img)

                vis = self.rnd_preview()
                msg.setIconPixmap(QPixmap(self.img_to_pixmap(vis)))

        except Exception as e:
            print(e)

        x = msg.exec_()
        print(x)
        while x == QMessageBox.Retry:
            vis = self.rnd_preview()
            msg.setIconPixmap(QPixmap(self.img_to_pixmap(vis)))
            x = msg.exec_()
            if x == QMessageBox.Close:
                break

    def rnd_preview(self):
        images = os.listdir(self.images_path)
        image = np.random.choice(images)
        func_to_exec = [np.random.randint(0, 2) for _ in range(len(self.function))]
        while sum(func_to_exec) < len(self.function) - 3:  # 7 func seçilirse en az 4ü çalışır.
            func_to_exec = [np.random.randint(0, 2) for _ in range(len(self.function))]
        orj_img = cv2.imread(self.images_path + image)
        cpy_img = orj_img.copy()
        cpy_mask = None
        for i in range(len(self.function)):
            if func_to_exec[i] == 1:
                cpy_img, cpy_mask = self.function[i](cpy_img, cpy_mask)
        w = 500
        h = 500
        orj_img = cv2.resize(orj_img, (w, h))
        cpy_img = cv2.resize(cpy_img, (w, h))
        # img_or = self.img_to_pixmap(orj_img)
        # img_aug = self.img_to_pixmap(cpy_img)
        vis = np.concatenate((orj_img, cpy_img), axis=1)
        return  vis
    def img_to_pixmap(self, cvImg):
        cvImg = cv2.cvtColor(cvImg, cv2.COLOR_RGB2BGR)
        height, width, channel = cvImg.shape
        bytesPerLine = 3 * width
        qImg = QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return qImg
    def progress_bar(self, value, endvalue):
        percent = float(value) / endvalue
        self.progressBar.setValue(int(round(percent * 100)))

    def mask_images_check(self):
        try:
            if self.check_mask_images.isChecked():
                self.isMask(True)
            else:
                self.isMask(False)
        except Exception as e:
            print(e)

    def isMask(self,boolValue):
        self.label_3.setEnabled(boolValue)
        self.label_4.setEnabled(boolValue)
        self.txt_mask_path.setEnabled(boolValue)
        self.txt_maskOut_path.setEnabled(boolValue)
        self.btn_mask_path.setEnabled(boolValue)
        self.btn_maskOut_path.setEnabled(boolValue)

    def btn_save_click(self):
        try:
            if self.check_mask_images.isChecked():
                if len(self.txt_mask_path.toPlainText()) <=0 or len(self.txt_img_path.toPlainText()) <=0:
                    self.listView.addItem("Mask Path or Image Path Not Selected")
                else:
                    self.set_save()
            else:
                if len(self.txt_img_path.toPlainText()) <=0:
                    self.listView.addItem("Image Path Not Selected")
                else:
                    self.set_save()
        except Exception as e:
            print(e)

    def set_save(self):
        if self.spin_expected_number.value() <= 0:
            try:
                self.expected_number = len(os.listdir(self.images_path))

            except Exception as e:
                print(e)
        else:
            self.expected_number = self.spin_expected_number.value()

        self.listView.addItem(" ____________________________________")
        self.listView.addItem(self.str_create("Ex. Number of Image: " + str(self.expected_number)))
        for each in self.listSettings:
            if each == "ROTATE":
                each = each + "{ MIN: " + str(self.spin_Rotate_min.value()) + " MAX: " + str(
                    self.spin_Rotate_max.value()) + "}"
            if each == "PAD":
                each = each + "{ MIN: " + str(self.spin_pad_min.value()) + " MAX: " + str(
                    self.spin_pad_max.value()) + "}"
            if each == "CONTRASST":
                each = each + "{ MIN: " + str(self.spin_Contrast_min.value()) + " MAX: " + str(
                    self.spin_Contrast_max.value()) + "}"
            if each == "BRİGHTESS":
                each = each + "{ MIN: " + str(self.spin_Brightness_min.value()) + " MAX: " + str(
                    self.spin_Brightness_max.value()) + "}"
            if each == "ZOOM":
                each = each + "{ MIN: " + str(self.spin_Zoom_min.value()) + " MAX: " + str(
                    self.spin_Zoom_max.value()) + "}"
            self.listView.addItem(self.str_create(each))
        self.btn_start.setEnabled(True)
        self.frame.setEnabled(False)
        self.check_mask_images.setEnabled(False)
        self.progressBar.setVisible(True)
        #self.progressBar.setFormat("%p/" + str(self.expected_number))
        self.listView.addItem("|____________________________________|")

    def str_create(self, txt):
        text = "|____________________________________|"
        list_text = list(text)
        list_txt = list(txt)
        l_text = len(text)
        l_txt = len(txt)
        index = int((l_text - l_txt) / 2)
        for each in range(len(txt)):
            list_text[index + each] = list_txt[each]

        str1 = ''.join(list_text)
        return str1

    def pad_check(self):
        self.Checkbox_change(self.check_pad, self.pad, "PAD")

    def Zoom_check(self):
        self.Checkbox_change(self.check_Zoom, self.zoom, "ZOOM")

    def Noisy_check(self):
        self.Checkbox_change(self.check_Noisy, self.noise, "NOISE")

    def fft2d_check(self):
        self.Checkbox_change(self.check_fft2d, self.fft2d, "FFT2D")

    def Brightess_check(self):
        self.Checkbox_change(self.check_Brightess, self.brightness, "BRİGHTESS")

    def Contrast_check(self):
        self.Checkbox_change(self.check_Contrast, self.contrast, "CONTRASST")

    def Rotate_check(self):
        self.Checkbox_change(self.check_Rotate, self.rotate, "ROTATE")

    def Checkbox_change(self, checkbox, func, funcName):
        if checkbox.isChecked():
            self.function.append(func)
            self.listSettings.append(funcName)
        else:
            self.function.remove(func)
            self.listSettings.remove(funcName)
        print(self.function)
        print(self.listSettings)

    def rotate(self, image, mask=None):
        angle = np.random.randint(self.spin_Rotate_min.value(), self.spin_Rotate_max.value())  # [-5, 6)
        (h, w) = image.shape[:2]
        rotation_point = (w / 2, h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(rotation_point, angle, 1.0)
        dst = cv2.warpAffine(image, rotation_matrix, (w, h))
        if mask is not None:
            last_chs = []
            for ch in range(mask.shape[2]):
                m = mask[:, :, ch]
                if ch != mask.shape[2] - 1:
                    m_dst = cv2.warpAffine(m, rotation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=[0, 0, 0])
                else:
                    m_dst = cv2.warpAffine(m, rotation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=[255, 255, 255])
                last_chs.append(m_dst)
            mask = cv2.merge(last_chs)
        return dst, mask

    def zoom(self, image, mask=None):
        scale = np.random.uniform(self.spin_Zoom_min.value(), self.spin_Zoom_max.value())  # [0.8, 1.2)
        (h, w) = image.shape[:2]
        rotation_point = (w / 2, h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(rotation_point, 0, scale)
        dst = cv2.warpAffine(image, rotation_matrix, (w, h))
        if mask is not None:
            last_chs = []
            for ch in range(mask.shape[2]):
                m = mask[:, :, ch]
                if ch != mask.shape[2] - 1:
                    m_dst = cv2.warpAffine(m, rotation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=[0, 0, 0])
                else:
                    m_dst = cv2.warpAffine(m, rotation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=[255, 255, 255])
                last_chs.append(m_dst)
            mask = cv2.merge(last_chs)
        return dst, mask

    def contrast(self, image, mask=None):
        clipLimit = random.randint(self.spin_Contrast_min.value(), self.spin_Contrast_max.value())
        rand_grid_size = random.randint(1, 8)
        tileGridSize = (rand_grid_size, rand_grid_size)
        clahe = cv2.createCLAHE(clipLimit, tileGridSize)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
        l, a, b = cv2.split(lab)  # split on 3 different channels
        l2 = clahe.apply(l)  # apply CLAHE to the L-channel
        lab = cv2.merge((l2, a, b))  # merge channels
        img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
        return img2, mask

    def noise(self, image, mask=None):
        noise_list = ["gauss", "s&p"]
        noise_typ = np.random.choice(noise_list)
        if noise_typ == "gauss":
            noisy = skimage.util.random_noise(image, mode='gaussian', seed=None, clip=True)
            noisy *= 255
            return noisy.astype("uint8"), mask
        elif noise_typ == "s&p":
            prob = np.random.uniform(0, 0.07)
            output = np.zeros(image.shape, np.uint8)
            thres = 1 - prob
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    rdn = np.random.random()
                    if rdn < prob:
                        output[i][j] = 0
                    elif rdn > thres:
                        output[i][j] = 255
                    else:
                        output[i][j] = image[i][j]
            return output, mask

    def brightness(self, img, mask=None):
        low, high = self.spin_Brightness_min.value(), self.spin_Brightness_max.value()
        value = random.uniform(low, high)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 1] = hsv[:, :, 1] * value
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        hsv[:, :, 2] = hsv[:, :, 2] * value
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img, mask

    def fft2d(self, image, mask=None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
        opt = np.random.randint(1, 4)
        coef = 50
        y_value_1, y_value_2 = np.random.randint(center_y // 6, center_y // 2.7), np.random.randint(center_y // 1.6,
                                                                                                    center_y // 1.1)
        y_value = np.random.choice([y_value_1, y_value_2])
        x_value_1, x_value_2 = np.random.randint(center_x // 10, center_x // 2.9), np.random.randint(center_x // 1.6,
                                                                                                     center_x // 1.18)
        x_value = np.random.choice([x_value_1, x_value_2])

        image_fft = np.fft.fft2(image)
        image = np.fft.fftshift(image_fft)
        if opt == 1:  # horizontal
            coef /= 10
            image[y_value, center_x] = image[y_value, center_x] * coef
            image[abs(image.shape[0] - y_value), center_x] = image[abs(image.shape[0] - y_value), center_x] * coef

        elif opt == 2:  # vertical
            coef /= 4
            image[center_y, x_value] = image[center_y, x_value] * coef
            image[center_y, abs(image.shape[1] - x_value)] = image[center_y, abs(image.shape[1] - x_value)] * coef

        elif opt == 3:  # eğik
            image[y_value, x_value] = image[y_value, x_value] * coef
            image[image.shape[0] - y_value, int(image.shape[1] - x_value)] = image[image.shape[0] - y_value, int(
                image.shape[1] - x_value)] * coef

        img_c4 = np.fft.ifftshift(image)
        img_c5 = np.fft.ifft2(img_c4)
        return np.abs(img_c5), mask

    def pad(self, image, mask=None):
        side = [0, 0, 0, 0]
        padding_width = np.random.uniform(self.spin_pad_min.value(), self.spin_pad_max.value())
        if np.random.randint(0, 2) == 1:
            color = [0, 0, 0]
        else:
            color = [255, 255, 255]
        while sum(side) < 2:
            side = [np.random.randint(0, 2), np.random.randint(0, 2), np.random.randint(0, 2), np.random.randint(0, 2)]
        borders = []
        border_height = int(padding_width * image.shape[0])  # shape[0] = rows
        border_width = int(padding_width * image.shape[1])  # shape[1] = cols
        for i, s in enumerate(side):
            if s == 1:
                if i < 2:
                    borders.append(border_height)
                else:
                    borders.append(border_width)
            else:
                borders.append(0)

        dst = cv2.copyMakeBorder(image, borders[0], borders[1], borders[2], borders[3], cv2.BORDER_CONSTANT, None,
                                 color)
        if mask is not None:
            last_chs = []
            for ch in range(mask.shape[2]):
                m = mask[:, :, ch]
                if ch != mask.shape[2] - 1:
                    m_dst = cv2.copyMakeBorder(m, borders[0], borders[1], borders[2], borders[3], cv2.BORDER_CONSTANT,
                                               None,
                                               [0, 0, 0])
                else:
                    m_dst = cv2.copyMakeBorder(m, borders[0], borders[1], borders[2], borders[3], cv2.BORDER_CONSTANT,
                                               None,
                                               [255, 255, 255])
                last_chs.append(m_dst)
            mask = cv2.merge(last_chs)
        return dst, mask

    def btn_maskOut_path_click(self):
        self.output_mask_path = self.setPath("Selected Mask Output Path", self.txt_maskOut_path)

    def btn_mask_path_click(self):
        self.mask_path = self.setPath("Selected Mask Path", self.txt_mask_path)

    def btn_imgOut_path_click(self):
        self.output_path = self.setPath("Selected Output Path",  self.txt_imgOut_path)

    def btn_img_path_click(self):
        self.images_path = self.setPath("Selected Image Path",  self.txt_img_path)

    def setPath(self, info, txtPath):
        try:
            input_dir = QFileDialog.getExistingDirectory(None, 'Select a folder:')
            if len(input_dir) > 0:
                strPath = input_dir + "/"
                txtPath.setPlainText(strPath)
                print(strPath)
                self.listView.addItem(info + ":\n   " + input_dir)
                return strPath
        except Exception as e:
            self.listView.addItem("Error: " + e)

if __name__ == "__main__":
    isMain = True
    app = QApplication(sys.argv)
    window = MainWindow()
    #pre = Mainpreview()
    window.show()
    sys.exit(app.exec_())