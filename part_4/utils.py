import pickle
import numpy as np


def read_txt_file(file_path):
    with open(file_path) as file:
        lines = [line.rstrip() for line in file]

    return lines


def read_pkl_file(pkl_path):
    with open(pkl_path, 'rb') as pklfile:
        data = pickle.load(pklfile, encoding='latin1')

    return data


def padding_img_3D(src_img):
    height, width, d = src_img.shape

    padded_img_3D = np.zeros((height + 81, width + 81, d), dtype='uint8')
    padded_img_3D[40:height + 40, 40:width + 40, :] = src_img

    return padded_img_3D


def crop_img_by_indx(src_img, indx):
    padded_img_3D = padding_img_3D(src_img)

    return padded_img_3D[indx[1]:indx[1] + 81, indx[0]:indx[0] + 81, :]


def crop_img_by_indx_list(img, indx_list):
    cropped = [crop_img_by_indx(img, indx) for indx in indx_list]
    return [x for x in cropped if x.shape == (81, 81, 3)]
