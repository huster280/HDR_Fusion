#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'gimbu'
__data__ = '30/03/17'
import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading


class Calibration(object):
    def __init__(self, gamma=10.0, LDR_SIZE=256):
        """
        :Description: to initial a Calibration instance
        :param gamma:  ldr_img lux level
        :param LDR_SIZE:  ldr_img lux level
        :return: Calibration instance
        """
        self.__gamma = gamma
        self.__LDR_SIZE = LDR_SIZE
        self.__intensity_weight_256x_ = self._gnrGaussianWeights(mu=127.5, sig=50)

    def _gnrGaussianWeights(self, mu=127.5, sig=50):
        """
        :Description: to generate a gaussian weights
        :param mu:  mu
        :param sig: sig
        :return w_256x: weights array
        """
        LDR_SIZE = self.__LDR_SIZE
        w_256x = np.zeros((LDR_SIZE, ), dtype="float32")
        for i in xrange(LDR_SIZE):
            # left = 1./(np.sqrt(2.*np.pi)*sig)
            left = 128
            right = np.exp(-(i - mu) * (i - mu) / (2 * sig * sig))
            w_256x[i] = left * right
        return w_256x

    def process(self, images, times, samples=70, random=False):
        """
        :Description: to calibrate CRF curve
        :param images: image list
        :param times: time list
        :param samples: samples point count
        :param random: whether samples random
        :return: response_256x1x3, CRF array
        """
        assert isinstance(images, list), 'images should be list'
        assert isinstance(times, list), 'times should be list'
        assert len(images) == len(times), "images length should be same as times"
        LDR_SIZE = self.__LDR_SIZE
        w = self.__intensity_weight_256x_.copy()
        gamma = self.__gamma
        images = np.array(images, dtype="uint8")
        times = np.array(times, dtype="float32")
        n_img = len(images)
        n_chn = images[0].shape[2]
        img_channel_list = []
        for i in xrange(n_chn):
            tmp = []
            for j in xrange(n_img):
                img_channel = cv2.split(images[j])[i]
                tmp.append(img_channel)
            img_channel_list.append(tmp)
        img_shape = images[0].shape
        img_cols = img_shape[1]
        img_rows = img_shape[0]
        sample_points_list = []

        # set random situation.
        if random is True:
            for i in xrange(samples):
                r = np.random.randint(0, img_rows)
                c = np.random.randint(0, img_cols)
                sample_points_list.append((r, c))
        if random is False:
            x_points = int(np.sqrt(samples * img_cols / img_rows))
            y_points = samples / x_points
            n_samples = x_points * y_points
            step_x = img_cols / x_points
            step_y = img_rows / y_points
            r = step_x / 2
            for j in xrange(y_points):
                rr = r + j * step_y
                c = step_y / 2
                for i in xrange(x_points):
                    cc = c + i * step_x
                    sample_points_list.append((rr, cc))

        # svd solve response curve.
        response_list = []
        for z in xrange(n_chn):
            eq = 0
            A = np.zeros((n_samples*n_img+LDR_SIZE+1, LDR_SIZE+n_samples), dtype="float32")
            B = np.zeros((A.shape[0]), dtype="float32")
            for i in xrange(n_samples):
                r = sample_points_list[i][0]
                c = sample_points_list[i][1]
                for j in xrange(n_img):
                    val = img_channel_list[z][j][r, c]
                    A[eq, val] = w[val]
                    A[eq, LDR_SIZE + i] = -w[val]
                    B[eq] = w[val] * np.log(times[j])
                    eq += 1
            # F(128)曝光量对数设0, 也就是曝光量为单位1, 不关事
            A[eq, LDR_SIZE / 2] = 1
            eq += 1
            for i in range(0, 254):
                A[eq, i] = gamma * w[i]
                A[eq, i+1] = -2 * gamma * w[i]
                A[eq, i+2] = gamma * w[i]
                eq += 1
            _, response = cv2.solve(A, B, flags=cv2.DECOMP_SVD)

            # just from ln(lum) convert to lum.
            response = cv2.exp(response)
            response_256x1 = response[:256]
            response_list.append(response_256x1)

        self.camera_response_256x1x3 = cv2.merge(response_list)
        # need return 256x1x3 nparray.
        return self.camera_response_256x1x3

    def showSaveData(self, filename):
        """
        :Description: showSaveData, show and save CRF curve datas
        :param filename: filename want to save CRF datas
        :return None:
        """
        response_256x1x3 = self.camera_response_256x1x3.copy()
        response_256x3 = response_256x1x3.reshape(256, 3)
        np.savetxt('./respose_curve/'+filename, response_256x3, fmt='%.2f',)
        _response_array = np.transpose(response_256x3)
        x = np.array(xrange(256))
        plt.figure(1)
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)
        plt.sca(ax1)
        plt.plot(x, _response_array[0], linewidth=2, color="b")
        plt.sca(ax2)
        plt.plot(x, _response_array[1], linewidth=2, color="g")
        plt.sca(ax3)
        plt.plot(x, _response_array[2], linewidth=2, color="r")
        plt.show()


class HdrMerge(object):
    def __init__(self,
                 gamma, contrast, saturation, sigma_space, sigma_color):
        """
        :Description: to initial a HdrMerge instance
        :param camera_response_256x1x3: CRF array
        :param gamma: Tonermap gamma
        :param contrast: Tonermap contrast
        :param saturation: Tonermap saturation
        :param sigma_space: Tonermap sigma_space
        :param sigma_color: Tonermap sigma_color
        :return: HdrMerge instance
        """
        self.__LDR_SIZE = 256
        # assert isinstance(camera_response, np.ndarray), 'camera_response should be np.array'
        # assert camera_response.shape[0] == self.__LDR_SIZE, 'camera_response should be right length'
        self.__intensity_weight_256x_ = self._gnrGaussianWeights(mu=127.5, sig=50)
        # self.__camera_response_256x1x3 = camera_response
        self.__gamma = gamma
        self.__contrast = contrast
        self.__saturation = saturation
        self.__sigma_space = sigma_space
        self.__sigma_color = sigma_color

    def preprocess(self, cali_gamma, LDR_SIZE, images, times):
        clb = Calibration(cali_gamma, LDR_SIZE)
        self.__camera_response_256x1x3 = clb.process(images, times)

    def _gnrGaussianWeights(self, mu=127.5, sig=50):
        """
        :Description: to generate a gaussian weights
        :param mu:  mu
        :param sig: sig
        :return w_256x: weights array
        """
        LDR_SIZE = self.__LDR_SIZE
        w_256x = np.zeros((LDR_SIZE, ), dtype="float32")
        for i in xrange(LDR_SIZE):
            # left = 1./(np.sqrt(2.*np.pi)*sig)
            left = 128
            right = np.exp(-(i - mu) * (i - mu) / (2 * sig * sig))
            w_256x[i] = left * right
        return w_256x

    def _merge(self, images, times):
        """
        :Description: use images, times, and CRF to merge HDRI
        :param images: image list
        :param times: times list
        :return hdr_img: HDRI(lux_img)
        """
        assert isinstance(images, list), 'images should be list'
        assert isinstance(times, list), 'times should be list'
        assert len(images) == len(times), "images length should be same as times"
        weights = self.__intensity_weight_256x_.copy()
        n_img = len(images)
        n_chn = images[0].shape[2]
        response_256x1x3 = self.__camera_response_256x1x3.copy()
        log_response = np.log(response_256x1x3)
        log_time = np.log(times)
        # log_hdr_img channel list
        hdr_chn_list = [0, 0, 0]
        img_avr_w_sum = np.zeros(images[0].shape[:2], dtype="float32")
        for i in xrange(n_img):
            src_chn_list = cv2.split(images[i])
            img_avr_w = np.zeros(images[0].shape[:2], dtype="float32")
            for cn in xrange(n_chn):
                img_cn_w = cv2.LUT(src_chn_list[cn], weights)
                img_avr_w += img_cn_w
            # 第n张图3个通道的平均权值图像
            img_avr_w /= n_chn
            # 一张图的log_response(log(lum))
            response_img = cv2.LUT(images[i], log_response)
            response_chn_list = cv2.split(response_img)
            for chn in xrange(n_chn):
                # img_avr_w:图片的平均通道权值 response_chn_list[chn]:通道的log_response log_time[i]:图片的log_time.
                hdr_chn_list[chn] += cv2.multiply(img_avr_w, response_chn_list[chn] - log_time[i])
                # 全部图的平均权值的和
            img_avr_w_sum += img_avr_w
        # 全部图的平均权值的和的倒数
        img_avr_w_sum = 1.0 / img_avr_w_sum
        for cn in xrange(n_chn):
            hdr_chn_list[cn] = cv2.multiply(hdr_chn_list[cn], img_avr_w_sum)
        log_hdr_img = cv2.merge(hdr_chn_list)
        # this is lux, 为什么和官方的数值有数量级的差别。
        hdr_img = cv2.exp(log_hdr_img)
        return hdr_img

    def _mapLuminance(self, src, lum, new_lum, saturation):
        """
        :Description: combine saturation weight to calculate new img
        :param src: BGR img
        :param lum: GRAY img
        :param new_lum: map img
        :param saturation: saturation
        :return new_img: new img
        """
        chn_list = cv2.split(src)
        for cn in xrange(len(chn_list)):
            chn_list[cn] = cv2.multiply(chn_list[cn], 1.0/lum)
            chn_list[cn] = cv2.pow(chn_list[cn], saturation)
            chn_list[cn] = cv2.multiply(chn_list[cn], new_lum)
        new_img = cv2.merge(chn_list)
        return new_img

    def process(self, images, times):
        """
        :Description: combine factors weight to merge HDRI and tonermap LDRI
        :param images: images list
        :param times: times list
        :return ldr_img: LDRI
        """
        assert isinstance(images, list), 'images should be list'
        assert isinstance(times, list), 'times should be list'
        assert len(images) == len(times), "images length should be same as times"
        gamma = self.__gamma
        contrast = self.__contrast
        saturation = self.__saturation
        sigma_space = self.__sigma_space
        sigma_color = self.__sigma_color

        hdr_img = self._merge(images, times)
        hdr_img_2d = hdr_img.reshape(images[0].shape[0], images[0].shape[1]*3)
        minval, maxvalue, _, _ = cv2.minMaxLoc(hdr_img_2d)
        img = (hdr_img - minval) / (maxvalue - minval)
        img = img.clip(1.0e-4)
        img = cv2.pow(img, 1.0 / gamma)

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        log_img = np.log(gray_img)
        map_img = cv2.bilateralFilter(log_img, -1, sigma_color, sigma_space)
        minval, maxval, _, _ = cv2.minMaxLoc(map_img)
        scale = contrast / (maxval - minval)
        map_img = cv2.exp(map_img * (scale - 1.0) + log_img)
        img = self._mapLuminance(img, gray_img, map_img, saturation)
        img = cv2.pow(img, 1.0 / gamma)
        # no problem!!
        img = img.clip(None, 1.0)
        img = img * 255
        ldr_img = img.astype("uint8")
        return ldr_img


# todo: optimize the algorithm
class HdrFusion(object):
    def __init__(self, wcon, wsat, wexp):
        """
        :Description: to initial a HdrFusion instance
        :param wcon: Fusion wcon
        :param wsat: Fusion wsat
        :param wexp: Fusion wexp
        :return: HdrFusion instance
        """
        self.wcon = wcon
        self.wsat = wsat
        self.wexp = wexp

    def normalize(self, images, i):
        images[i] = images[i].astype("float32")
        images[i] /= 255.0

    # solve contrast and wellexp
    def sovle_c_w(self, contrast, wellexp, images, shape, i, is_gray):
        if is_gray:
            contrast[i] = cv2.Laplacian(images[i][:, :, 0], cv2.CV_32F)
        else:
            gray = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
            contrast[i] = cv2.Laplacian(gray, cv2.CV_32F)
        contrast[i] = np.abs(contrast[i])
        if self.wcon != 1:
            contrast[i] = cv2.pow(contrast[i], self.wcon)

        wellexp[i] = np.ones(shape, dtype="float32")
        if self.wexp != 0:
            splitted = [images[i][:, :, 0], images[i][:, :, 1], images[i][:, :, 2]]
            for img_cn in splitted:
                expo = cv2.subtract(img_cn, 0.5, dtype=cv2.CV_32F)
                expo = cv2.pow(expo, 2.0)
                expo = -expo / 0.08
                # larger '0.08' only make 'cv2.exp(expo)' nearest "1"
                expo = cv2.exp(expo)
                wellexp[i] *= expo
            wellexp[i] = cv2.pow(wellexp[i], self.wexp)

    # @profile
    def process(self, images):
        """
        :Description: combine factors weight to fusion HDRI-alternate
        :param images: img list
        :return fusion_img: fusion_img
        """
        assert isinstance(images, list), 'images should be list'
        assert images[0].shape[-1] == 3, 'the input images should have 3 channels'
        n_img = len(images)
        # camera images both are [b, g, r] 3channels
        n_chn = images[0].shape[2]  # n_chn = 3

        rows, cols = images[0].shape[0], images[0].shape[1]
        shape = (rows, cols)

        # initial the weights[]
        weights = [None] * n_img
        weight_sum = np.zeros(shape, dtype="float32")

        # normalize img make convergence speedup
        # for i in xrange(n_img):
        #     images[i] = images[i].astype("float32")
        #     images[i] /= 255.0

        # use multithreading to normalize images
        threads = []
        nloops = xrange(n_img)
        # 完成所有线程分配，并不立即开始执行
        for i in nloops:
            t = threading.Thread(target=self.normalize, args=(images, i,))
            threads.append(t)
        # 开始调用start方法，同时开始所有线程
        for i in nloops:
            threads[i].start()
        # join方法：主线程等待所有子线程执行完成，再执行主线程接下来的操作。
        for i in nloops:
            threads[i].join()

        # 若输入为灰度图,饱和度saturation=0, weight 等于 contrast * wellexp
        is_gray = (images[0][:, :, 0] == images[0][:, :, 1]).all() and (images[0][:, :, 1] == images[0][:, :, 2]).all()

        # solve each image weight and solve weight_sum
        contrast = [None] * n_img
        wellexp = [None] * n_img
        threads = []
        for i in nloops:
            t = threading.Thread(target=self.sovle_c_w, args=(contrast, wellexp, images, shape, i, is_gray))
            threads.append(t)
        for i in nloops:
            threads[i].start()
        for i in nloops:
            threads[i].join()

        if is_gray:
            for i in xrange(n_img):
                weights[i] = contrast[i]
                if self.wexp != 0:
                    weights[i] *= wellexp[i]

                weights[i] += 1e-12
                weight_sum += weights[i]
        else:
            for i in xrange(n_img):
                img = images[i]

                mean = np.zeros(shape, dtype="float32")
                splitted = [img[:, :, 0], img[:, :, 1], img[:, :, 2]]  # 不用split，改用numpy矩阵索引
                for img_cn in splitted:
                    mean += img_cn
                mean /= n_chn

                # solve saturation
                saturation = np.zeros(shape, dtype="float32")
                for img_cn in splitted:
                    deviation = cv2.pow(img_cn - mean, 2.0)
                    saturation += deviation
                saturation = cv2.sqrt(saturation)
                # pow respective ratio
                saturation = cv2.pow(saturation, self.wsat)

                weight = contrast[i]
                weight *= saturation
                weight *= wellexp[i]
                weight += 1e-12
                weights[i] = weight
                weight_sum += weight

        maxlevel = int(np.log2(min(rows, cols)))
        # (maxlevel+1) images, following to solve the final pyramid.
        res_pyr = [None] * (maxlevel + 1)

        for i in xrange(n_img):
            img_pyr = [None] * (maxlevel + 1)
            weight_pyr = [None] * (maxlevel + 1)
            weights[i] /= weight_sum
            weight_pyr[0] = weights[i]
            if is_gray:
                img_pyr[0] = images[i][:, :, 0]  # 灰度图只计算img_pyr单通道
            else:
                img_pyr[0] = images[i]

            # following: buildPyramid(img, img_pyr, maxlevel)
            #            buildPyramid(weights[i], weight_pyr, maxlevel)
            for lvl in xrange(maxlevel):
                img_pyr[lvl + 1] = cv2.pyrDown(img_pyr[lvl])
                weight_pyr[lvl + 1] = cv2.pyrDown(weight_pyr[lvl])

            for lvl in xrange(maxlevel):
                # size = width, height
                size = img_pyr[lvl].shape[:2][::-1]
                up = cv2.pyrUp(img_pyr[lvl + 1], dstsize=size)
                img_pyr[lvl] -= up      # 减去上一层的扩展：拉普拉斯金字塔,得到边缘图,最上层不变

            for lvl in xrange(maxlevel + 1):
                if is_gray:
                    img_pyr[lvl] *= weight_pyr[lvl]
                else:
                    weight_list = [weight_pyr[lvl]] * 3
                    weight_merged = cv2.merge(weight_list)
                    img_pyr[lvl] *= weight_merged

                # first image to assign res_pry[0-maxlevel]
                if i == 0:
                    res_pyr[lvl] = img_pyr[lvl]
                # latter image to add to res_pry[0-maxlevel]
                else:
                    res_pyr[lvl] += img_pyr[lvl]       # img_pyr -> res_pyr,边缘(细节)增多,再加上平坦区域得到原图

        # 第一层求出第0层在第一次loop就已经完成了，后面的loop没有意义
        for lvl in xrange(maxlevel, 0, -1):
            # size = width, height
            size = res_pyr[lvl - 1].shape[:2][::-1]
            up = cv2.pyrUp(res_pyr[lvl], dstsize=size)
            res_pyr[lvl - 1] += up

        dst_tmp = res_pyr[0]
        dst_tmp *= 255
        dst_tmp[dst_tmp < 0] = 0
        dst_tmp[dst_tmp > 255] = 255  # 最大值设为255

        if is_gray:     # dst_tmp为单通道
            dst_tmp = dst_tmp.astype("uint8")
            dst_list = [dst_tmp] * 3
            fusion_img = cv2.merge(dst_list)  # return a 3-channal image
        else:
            fusion_img = dst_tmp.astype("uint8")

        return fusion_img
