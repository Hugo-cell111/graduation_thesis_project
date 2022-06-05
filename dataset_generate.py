'''
for Crack500
import cv2
import math
import glob
import random
import argparse

import numpy as np

from data.augmentation import*

def main():
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_val_data_dir",
                        type = str,
                        required = True,
                        help = "The directory of training/validation dataset(not label)")
    parser.add_argument("--test_data_dir",
                        type = str,
                        required = True,
                        help = "The directory of testing dataset(not label)")
    parser.add_argument("--train_val_dataset_name",
                        type = str,
                        required = True,
                        help = "The name of training/validation dataset")
    parser.add_argument("--test_dataset_name",
                        type = str,
                        required = True,
                        help = "The name of training/validation dataset")
    parser.add_argument("--train_dataset_ratio",
                        type = float,
                        required = True,
                        help = "train dataset vs validation dataset")
    parser.add_argument("--target1",
                        type = str,
                        required = True,
                        help = "whether train_val or test")
    parser.add_argument("--threshold",
                        type = float,
                        required = True)
    parser.add_argument("--patch_size",
                        type = int,
                        required = True)
    args = parser.parse_args()
    if args.target1 == "train_val":
        train_val_imagelist = glob.glob(args.train_val_data_dir + "*.jpg")
        train_val_num = len(train_val_imagelist)
        train_num = math.floor(train_val_num * args.train_dataset_ratio)
        # validation dataset
        f = open("./datasets/val/val_example.txt", "w+")
        for i in range(train_num, train_val_num):
            img_name = train_val_imagelist[i]
            gt_name = img_name.replace('.jpg', '_mask.png')
            img, mask = cv2.imread(img_name), cv2.imread(gt_name)
            if len(mask.shape) != 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            k = 1
            for i in range(4):
                for j in range(4):
                    img1 = img[i*360: (i+1)*360, j*640: (j+1)*640]
                    mask1 = mask[i*360: (i+1)*360, j*640: (j+1)*640]
                    mask1[mask1 <= args.threshold * 255] = 0
                    mask1[mask1 >= args.threshold * 255] = 255
                    filepath1 = img_name.replace(args.train_val_dataset_name, "val").replace(".jpg", "_" + str(k) + ".jpg")
                    filepath2 = gt_name.replace(args.train_val_dataset_name, "val").replace(".png", "_" + str(k) + ".png")
                    cv2.imwrite(filepath1, img1)
                    cv2.imwrite(filepath2, mask1)
                    f.write(filepath1 + " " + filepath2 + "\n")
                    k += 1
        f.close()
        # training dataset
        f = open("./datasets/train/train_example.txt", "w+")
        for item in range(train_num):
            print(item)
            img_name = train_val_imagelist[item]
            gt_name = img_name.replace('.jpg', '_mask.png')
            img, mask = cv2.imread(img_name), cv2.imread(gt_name)
            if len(mask.shape) != 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            k = 1
            for i in range(4):
                for j in range(4):
                    img1 = img[i * 360: (i + 1) * 360, j * 640: (j + 1) * 640]
                    mask1 = mask[i * 360: (i + 1) * 360, j * 640: (j + 1) * 640]
                    mask1[mask1 <= args.threshold * 255] = 0
                    mask1[mask1 >= args.threshold * 255] = 255
                    filepath1 = img_name.replace(args.train_val_dataset_name, "train").replace(".jpg", "_" + str(k) + ".jpg")
                    filepath2 = gt_name.replace(args.train_val_dataset_name, "train").replace("_mask.png", "_" + str(k) + "_mask.png")
                    if mask1.sum(axis=1).sum(axis=0) > 1000:

                        # origin
                        cv2.imwrite(filepath1, img1)
                        cv2.imwrite(filepath2, mask1)
                        f.write(filepath1 + " " + filepath2 + "\n")

                        # rotate/flip(total1. Change mask)
                        rand_num = random.randint(0, 2)
                        if rand_num == 2:
                            img2, mask2 = np.rot90(img1, 2), np.rot90(mask1, 2)
                        else:
                            img2, mask2 = cv2.flip(img1, rand_num), cv2.flip(mask1, rand_num)
                        filepath3 = filepath1.replace(".jpg", "_rf.jpg")
                        filepath4 = filepath2.replace("_mask.png", "_rf_mask.png")
                        cv2.imwrite(filepath3, img2)
                        cv2.imwrite(filepath4, mask2)
                        f.write(filepath3 + " " + filepath4 + "\n")

                        # noise(total1. Do not change mask)
                        a = random.random()
                        if a <= 0.5:
                            img2 = sp_noise(image=img1, prob=0.05)
                        else:
                            img2 = cv2.blur(img1, (5,5))
                        filepath3 = filepath1.replace(".jpg", "_noise.jpg")
                        filepath4 = filepath2.replace("_mask.png", "_noise_mask.png")
                        cv2.imwrite(filepath3, img2)
                        cv2.imwrite(filepath4, mask1)
                        f.write(filepath3 + " " + filepath4 + "\n")

                        # contrast(total1. Do not change mask)
                        img2 = img1 / 255.0  # 注意255.0得采用浮点数
                        img2 = np.power(img2, 0.4) * 255.0
                        img2 = img2.astype(np.uint8)
                        filepath3 = filepath1.replace(".jpg", "_contrast.jpg")
                        filepath4 = filepath2.replace("_mask.png", "_contrast_mask.png")
                        cv2.imwrite(filepath3, img2)
                        cv2.imwrite(filepath4, mask1)
                        f.write(filepath3 + " " + filepath4 + "\n")
                    k += 1
        f.close()

    elif args.target1 == "test":
        test_imagelist = glob.glob(args.test_data_dir + "*.jpg")
        test_num = len(test_imagelist)
        # test
        f = open("./datasets/test/test_example.txt", "w+")
        for i in range(test_num):
            img_name = test_imagelist[i]
            #gt_name = img_name.replace("img", "gt").replace('png', 'bmp')
            gt_name = img_name.replace('jpg', '_mask.png')
            ''''''
            img, mask = cv2.imread(img_name), cv2.imread(gt_name)
            if len(mask.shape) != 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            height, width = img.shape[0], img.shape[1]
            height -= height % 32
            width -= width % 32
            img = cv2.resize(img, (height, width), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (height, width), interpolation=cv2.INTER_LINEAR)
            mask[mask <= args.threshold * 255] = 0
            mask[mask >= args.threshold * 255] = 255
            filepath1 = img_name.replace(args.test_dataset_name, "test")
            filepath2 = gt_name.replace(args.test_dataset_name, "test")
            cv2.imwrite(filepath1, img)
            cv2.imwrite(filepath2, mask)
            ''''''
            f.write(img_name + " " + gt_name + "\n")
        f.close()

if __name__ == '__main__':
    main()
'''

import cv2
import math
import glob
import random
import argparse
from data.augmentation import *


def main():
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_val_data_dir",
                        type=str,
                        required=True,
                        help="The directory of training/validation dataset(not label)")
    parser.add_argument("--test_data_dir",
                        type=str,
                        required=True,
                        help="The directory of testing dataset(not label)")
    parser.add_argument("--train_val_dataset_name",
                        type=str,
                        required=True,
                        help="The name of training/validation dataset")
    parser.add_argument("--test_dataset_name",
                        type=str,
                        required=True,
                        help="The name of training/validation dataset")
    parser.add_argument("--train_dataset_ratio",
                        type=float,
                        required=True,
                        help="train dataset vs validation dataset")
    parser.add_argument("--target1",
                        type=str,
                        required=True,
                        help="whether train_val or test")
    parser.add_argument("--threshold",
                        type=float,
                        required=True)
    parser.add_argument("--patch_size",
                        type=int,
                        required=True)
    args = parser.parse_args()
    if args.target1 == "train_val":
        train_val_imagelist = glob.glob(args.train_val_data_dir + "*.jpg")
        train_val_num = len(train_val_imagelist)
        train_num = math.floor(train_val_num * args.train_dataset_ratio)
        # validation dataset
        f = open("./datasets/val/val_example.txt", "w+")
        for i in range(train_num, train_val_num):
            img_name = train_val_imagelist[i]
            gt_name = img_name.replace("img", "gt").replace('jpg', 'bmp')
            img, mask = cv2.imread(img_name), cv2.imread(gt_name)
            if len(mask.shape) != 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            height, width = img.shape[0], img.shape[1]
            height -= height % 32
            width -= width % 32
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
            mask[mask <= args.threshold * 255] = 0
            mask[mask >= args.threshold * 255] = 255
            filepath1 = img_name.replace(args.train_val_dataset_name, "val")
            filepath2 = gt_name.replace(args.train_val_dataset_name, "val")
            cv2.imwrite(filepath1, img)
            cv2.imwrite(filepath2, mask)
            f.write(filepath1 + " " + filepath2 + "\r\n")
        f.close()
        num_crop = 9
        # training dataset
        f = open("./datasets/train/train_example.txt", "w+")
        for i in range(train_num):
            print(i)
            img_name = train_val_imagelist[i]
            gt_name = img_name.replace("img", "gt").replace('jpg', 'bmp')
            img, mask = cv2.imread(img_name), cv2.imread(gt_name)
            if len(mask.shape) != 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            height, width = img.shape[0], img.shape[1]
            for j in range(num_crop):
                a = random.randint(0, height - args.patch_size - 1)
                b = random.randint(0, width - args.patch_size - 1)
                img_crop = img[a: a + args.patch_size, b: b + args.patch_size]
                mask_crop = mask[a: a + args.patch_size, b: b + args.patch_size]
                if mask_crop.sum(axis=1).sum(axis=0) > 1000:
                    # origin
                    filepath1 = img_name.replace(args.train_val_dataset_name, "train").replace(".jpg", "_crop" +
                                                                                               str(j) + ".jpg")
                    filepath2 = gt_name.replace(args.train_val_dataset_name, "train").replace(".bmp", "_crop" +
                                                                                               str(j) + ".bmp")
                    cv2.imwrite(filepath1, img_crop)
                    cv2.imwrite(filepath2, mask_crop)
                    f.write(filepath1 + " " + filepath2 + "\r\n")
                    # rotate(total3. Change mask)
                    for j1 in range(1, 4):
                        img1, mask1 = np.rot90(img_crop, j1), np.rot90(mask_crop, j1)
                        filepath1 = img_name.replace(args.train_val_dataset_name, "train").replace(".jpg", "_crop"
                                                                                                   + str(j) + "_rot" + str(
                            90 * j1) + ".jpg")
                        filepath2 = gt_name.replace(args.train_val_dataset_name, "train").replace(".bmp", "_crop"
                                                                                                  + str(j) + "_rot" + str(
                            90 * j1) + ".bmp")
                        cv2.imwrite(filepath1, img1)
                        cv2.imwrite(filepath2, mask1)
                        f.write(filepath1 + " " + filepath2 + "\r\n")
                    # flip(total2. Change mask)
                    for j2 in range(2):
                        img1, mask1 = cv2.flip(img_crop, j2), cv2.flip(mask_crop, j2)
                        filepath1 = img_name.replace(args.train_val_dataset_name, "train").replace(".jpg", "_crop"
                                                                                                   + str(j) + "_flip" + str(
                            j2) + ".jpg")
                        filepath2 = gt_name.replace(args.train_val_dataset_name, "train").replace(".bmp", "_crop"
                                                                                                  + str(j) + "_flip" + str(
                            j2) + ".bmp")
                        cv2.imwrite(filepath1, img1)
                        cv2.imwrite(filepath2, mask1)
                        f.write(filepath1 + " " + filepath2 + "\r\n")
                    '''''''''
                    # elastic_transform(total1. Change mask)
                    mask_crop = mask_crop.reshape(mask_crop.shape[0], mask_crop.shape[1], 1)
                    mask1 = np.concatenate((mask_crop, mask_crop, mask_crop), axis=2)
                    im_merge = np.concatenate((img_crop, mask1), axis=2)
                    im_merge = elastic_transform(image=im_merge, alpha=mask.shape[1] * 2,
                                                 sigma=mask.shape[1] * 0.08,
                                                 alpha_affine=mask.shape[1] * 0.08)
                    img1 = im_merge[..., :3]
                    mask1 = im_merge[..., 3]
                    mask1[mask1 <= args.threshold * 255] = 0
                    mask1[mask1 >= args.threshold * 255] = 255
                    filepath1 = img_name.replace(args.train_val_dataset_name, "train").replace(".jpg", "_crop"
                                                                                               + str(j) + "_ela.jpg")
                    filepath2 = gt_name.replace(args.train_val_dataset_name, "train").replace(".bmp", "_crop"
                                                                                              + str(j) + "_ela.bmp")
                    cv2.imwrite(filepath1, img1)
                    cv2.imwrite(filepath2, mask1)
                    f.write(filepath1 + " " + filepath2 + "\r\n")
                    '''''''''
                    # noise(total3. Do not change mask)
                    img1 = sp_noise(image=img_crop, prob=0.1)
                    filepath1 = img_name.replace(args.train_val_dataset_name, "train").replace(".jpg", "_crop"
                                                                                               + str(j) + "_spnoise.jpg")
                    filepath2 = gt_name.replace(args.train_val_dataset_name, "train").replace(".bmp", "_crop"
                                                                                              + str(j) + "_spnoise.bmp")
                    cv2.imwrite(filepath1, img1)
                    cv2.imwrite(filepath2, mask_crop)
                    f.write(filepath1 + " " + filepath2 + "\r\n")

                    img1 = gasuss_noise(image=img_crop)
                    filepath1 = img_name.replace(args.train_val_dataset_name, "train").replace(".jpg", "_crop"
                                                                                               + str(
                        j) + "_gasussnoise.jpg")
                    filepath2 = gt_name.replace(args.train_val_dataset_name, "train").replace(".bmp", "_crop"
                                                                                              + str(j) + "_gasussnoise.bmp")
                    cv2.imwrite(filepath1, img1)
                    cv2.imwrite(filepath2, mask_crop)
                    f.write(filepath1 + " " + filepath2 + "\r\n")

                    img1 = cv2.GaussianBlur(img_crop, (9, 9),0)
                    filepath1 = img_name.replace(args.train_val_dataset_name, "train").replace(".jpg", "_crop"
                                                                                               + str(
                        j) + "_gasussblur.jpg")
                    filepath2 = gt_name.replace(args.train_val_dataset_name, "train").replace(".bmp", "_crop"
                                                                                              + str(j) + "_gasussblur.bmp")
                    cv2.imwrite(filepath1, img1)
                    cv2.imwrite(filepath2, mask_crop)
                    f.write(filepath1 + " " + filepath2 + "\r\n")

                    # contrast(total2. Do not change mask)
                    img1 = img_crop / 255.0  # 注意255.0得采用浮点数
                    img1 = np.power(img1, 0.4) * 255.0
                    img1 = img1.astype(np.uint8)
                    img2 = img_crop / 255.0
                    img2 = np.power(img2, 1.2) * 255.0
                    img2 = img2.astype(np.uint8)
                    filepath1 = img_name.replace(args.train_val_dataset_name, "train").replace(".jpg",  "_crop"
                                                                                               + str(j) + "_contrast1.jpg")

                    filepath2 = gt_name.replace(args.train_val_dataset_name, "train").replace(".bmp",  "_crop"
                                                                                               + str(j) + "_contrast1.bmp")
                    filepath3 = img_name.replace(args.train_val_dataset_name, "train").replace(".jpg",  "_crop"
                                                                                               + str(j) + "_contrast2.jpg")
                    filepath4 = gt_name.replace(args.train_val_dataset_name, "train").replace(".bmp",  "_crop"
                                                                                               + str(j) + "_contrast2.bmp")
                    cv2.imwrite(filepath1, img1)
                    cv2.imwrite(filepath2, mask_crop)
                    f.write(filepath1 + " " + filepath2 + "\r\n")
                    cv2.imwrite(filepath3, img2)
                    cv2.imwrite(filepath4, mask_crop)
                    f.write(filepath3 + " " + filepath4 + "\r\n")
        f.close()

    elif args.target1 == "test":
        test_imagelist = glob.glob(args.test_data_dir + "*.jpg")
        test_num = len(test_imagelist)
        # test
        f = open("./datasets/test/test_example.txt", "w+")
        for i in range(test_num):
            img_name = test_imagelist[i]
            # gt_name = img_name.replace("img", "gt").replace('png', 'bmp')
            gt_name = img_name.replace('jpg', '_mask.png')
            f.write(img_name + " " + gt_name + "\n")
        f.close()


if __name__ == '__main__':
    main()