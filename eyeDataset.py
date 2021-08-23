import random
from shutil import copyfile

from torch.utils.data import Dataset
import os
import json
import glob
from PIL import Image
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
validation_split = 0.2
random_seed = 250
import cv2

def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img
class CommonEyeDataset(Dataset):
    def __init__(self,dir = "/data1/lulixian/eyeData/PM6400",class_name=["PM","NotPM"]):
        self.transform = transforms.Compose([

            transforms.Resize((224, 224)),
            # transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.35, 0.22, 0.10), std=(0.28, 0.18, 0.13)),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        self.data = glob.glob(dir + "/**/*.[jJ][pP][gG]",recursive=True) \
                    + glob.glob(dir + "/**/*.[jJ][pP][eE][gG]",recursive=True)
        self.label = [class_name.index(s.split("/")[-2]) for s in self.data]
    def __getitem__(self, item):
        img = cv2.imread(self.data[item])
        if img is None:
            print(self.data[item])
        img = img[:, int(img.shape[1] / 2 - img.shape[0] / 2):int(img.shape[1] / 2 + img.shape[0] / 2), :]
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = self.transform(img)
        return img, self.label[item],self.data[item]
    def __len__(self):
        return len(self.data)
class PredictEyeDataset(Dataset):
    def __init__(self, dir="final_mix"):
        transform = transforms.Compose([

            transforms.Resize((224, 224)),
            # transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.35, 0.22, 0.10), std=(0.28, 0.18, 0.13)),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        self.transform = transform
        # self.data = glob.glob(dir+"/**/*.[jJ][pP][gG]",recursive=True) + glob.glob(dir+"/**/*.[jJ][pP][eE][gG]",recursive=True)
        self.data = glob.glob(dir + "/*.[jJ][pP][gG]") + glob.glob(dir + "/*.[jJ][pP][eE][gG]")
        print(self.data)
    def __getitem__(self, item):
        img = cv2.imread(self.data[item])
        if img is None:
            print(self.data[item])

        img = img[:, int(img.shape[1]/2 - img.shape[0]/2):int(img.shape[1]/2 + img.shape[0]/2),:]
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = self.transform(img)
        return img,self.data[item]
    def __len__(self):
        return len(self.data)

class EyeDataset(Dataset):

    def __init__(self,transform=None,train=0,data_version = "final_mix"):
        if transform == None:
            transform = transforms.Compose([

                transforms.Resize((224, 224)),
                # transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.35, 0.22, 0.10), std=(0.28, 0.18, 0.13)),
                # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ])
        if transform == -1:
            transform = None
        self.transform = transform
        self.all_data = []#self.data3
        self.all_label = []
        for k in range(5):
            # if k > 0:
            #     continue
            for i in glob.glob(f"/data1/lulixian/eyeData/{data_version}/{k}/*"):
                self.all_data.append(i)
                self.all_label.append(k)
        # indices = list(range(len(self.all_data)))
        # split = int(np.floor(validation_split * len(self.all_data)))
        # np.random.seed(random_seed)
        # np.random.shuffle(indices)
        # train_indices,valid_indices = indices[split:],indices[:split]
        x_train,x_test_val,y_train,y_test_val = train_test_split(self.all_data,self.all_label,test_size = 0.4,
                                                         random_state = random_seed,stratify=self.all_label)
        x_test,x_val,y_test,y_val = train_test_split(x_test_val,y_test_val,test_size = 0.5,
                                                         random_state = random_seed,stratify=y_test_val)
        if train == 0:
            self.data = x_train
            self.label = y_train
        elif train == 1:
            self.data = x_val
            self.label = y_val
        elif train == 2:
            self.data = x_test
            self.label = y_test
        from collections import Counter
        a = [0]*5
        for i in range(5):
            a[i] = self.label.count(i)
        weights = []
        for i in self.label:
            weights.append(1.0/a[i])
        self.weights = weights
        self.c = 5
    def save_val(self):
        os.makedirs("val",exist_ok=True)
        for i in range(5):
            os.makedirs(f"val/{i}",exist_ok=True)
        for i in range(len(self.data)):
            img_path = self.data[i]
            copyfile(img_path, f"val/{self.label[i]}/{img_path.split('/')[-1]}")

    def cal_mean(self,num = 200):
        imgs = np.zeros([224, 224, 3, 1])
        means, stdevs = [], []
        random.shuffle(self.data)
        for i in range(num):
            img = cv2.imread(self.data[i])

            img = img[:, int(img.shape[1] / 2 - img.shape[0] / 2):int(img.shape[1] / 2 + img.shape[0] / 2), :]
            img = cv2.resize(img, (224, 224))
            img = img[:, :, :, np.newaxis]
            imgs = np.concatenate((imgs, img), axis=3)

        imgs = imgs.astype(np.float32) / 255.
        for i in range(3):
            # pixels = imgs[:, :, i, :].ravel()  # 拉成一行
            means.append(np.mean(imgs[:, :, i, :]))
            stdevs.append(np.std(imgs[:, :, i, :]))
        means.reverse()
        stdevs.reverse()
        print("normMean = {}".format(means))
        print("normStd = {}".format(stdevs))
    def show(self,item):
        # img = cv2.imread(self.data[item])
        # if img is None:
        #     print(self.data[item])
        # img = img[:, int(img.shape[1] / 2 - img.shape[0] / 2):int(img.shape[1] / 2 + img.shape[0] / 2), :]
        # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = Image.open(self.data[item])
        transform = transforms.Compose([

            transforms.Resize((224, 224)),
            # transforms.RandomCrop((224, 224)),
            # transforms.Normalize(mean=(0.35, 0.22, 0.10), std=(0.28, 0.18, 0.13)),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        img = transform(img)
        plt.figure("Image")  # 图像窗口名称
        plt.imshow(img)
        plt.axis('on')  # 关掉坐标轴为 off
        plt.title('image')  # 图像题目
        plt.show()

    def show2(self, item,brightness=0.8,contrast=1,saturation=0):
        img = Image.open(self.data[item])
        transform = transforms.Compose([

            transforms.Resize((224, 224)),
            # transforms.RandomCrop((224, 224)),
            transforms.ColorJitter(brightness=[brightness, brightness], contrast=[contrast, contrast], saturation=saturation, hue=0)

            # transforms.Normalize(mean=(0.35, 0.22, 0.10), std=(0.28, 0.18, 0.13)),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        img = transform(img)
        plt.figure("Image")  # 图像窗口名称
        plt.imshow(img)
        plt.axis('on')  # 关掉坐标轴为 off
        plt.title('image')  # 图像题目
        plt.show()
    def __getitem__(self, item):

        img = cv2.imread(self.data[item])
        if img is None:
            print(self.data[item])
        # img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        # np.clip(img_gray,0,100)
        # img_gray = img_gray % 100
        #
        # cv2.imshow("img", img_gray)
        # cv2.waitKey(0)

        #
        #
        # height = [0] * 2
        # for i in range(img.shape[1]):
        #     if not np.all(img[:,i] == 0):
        #         height[0] = i
        #         break
        # for i in reversed(range(img.shape[1])):
        #     if not np.all(img[:,i] == 0):
        #         height[1] = i
        #         break
        # print(width)
        # print(height)
        # print(img.shape)
        # img = img[width[0]:width[1],height[0]:height[1],:]
        img = img[:, int(img.shape[1]/2 - img.shape[0]/2):int(img.shape[1]/2 + img.shape[0]/2),:]
        # print(img.shape)
        # img = cv2.resize(img,(512,512),interpolation = cv2.INTER_AREA)
        #
        # cv2.imshow("img",img)
        # cv2.waitKey(0)

        # img = hisEqulColor(img)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # img = Image.open(self.data[item])


        # img = plt.imread(os.path.join(self.img_path,fn))

        if self.transform:
            img = self.transform(img)
        # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        #
        # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img,self.label[item]

    def __len__(self):
        return len(self.data)
class PMDataset(EyeDataset):
    def __init__(self,transform=None,train=0):
        if transform == None:
            transform = transforms.Compose([

                transforms.Resize((224, 224)),
                # transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.35, 0.22, 0.10), std=(0.28, 0.18, 0.13)),
                # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ])
        self.transform = transform
        self.all_data = []
        self.all_label = []
        for k in range(5):
            # if k > 0:
            #     continue
            for i in glob.glob(f"/data1/lulixian/eyeData/final_mix/{k}/*"):
                if k <= 1:
                    self.all_data.append(i)
                    self.all_label.append(0) #notPM
                else:
                    self.all_data.append(i)
                    self.all_label.append(1)  # notPM
        for i in glob.glob(f"/data1/lulixian/eyeData/final_mix/NotPM/*"):
            self.all_data.append(i)
            self.all_label.append(0)  # notPM
        for i in glob.glob(f"/data1/lulixian/eyeData/final_mix/PM/*"):
            self.all_data.append(i)
            self.all_label.append(1)  # PM
        x_train, x_test_val, y_train, y_test_val = train_test_split(self.all_data, self.all_label, test_size=0.4,
                                                                    random_state=random_seed, stratify=self.all_label)
        x_test, x_val, y_test, y_val = train_test_split(x_test_val, y_test_val, test_size=0.5,
                                                        random_state=random_seed, stratify=y_test_val)
        if train == 0:
            self.data = x_train
            self.label = y_train
        elif train == 1:
            self.data = x_val
            self.label = y_val
        elif train == 2:
            self.data = x_test
            self.label = y_test
        a = [0] * 2
        for i in range(2):
            a[i] = self.label.count(i)
        weights = []
        for i in self.label:
            weights.append(1.0 / a[i])
        self.weights = weights
class EyeDataset_two_divide(EyeDataset):

    def __init__(self,transform=None,train=0,data_version = "final_mix",class_type = 0):
        # dir = "json"
        # json_name = ""
        # self.img_path = ""
        # if train:
        #     json_name = "set1.json"
        #     self.img_path = "data/set1"
        # else:
        #     json_name = "set2.json"
        #     self.img_path = "data/set2"
        if transform == None:
            transform = transforms.Compose([

                transforms.Resize((224, 224)),
                # transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.35, 0.22, 0.10), std=(0.28, 0.18, 0.13)),
                # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ])
        if transform == -1:
            transform = None
        self.transform = transform

        # with open("json/set1.json", 'r', encoding='utf-8') as json_file:
        #     self.data1 =[(os.path.join("data/set1",v['filename']),self.trans_label(v['image_labels'])) for k,v in json.load(json_file).items()]
        # with open("json/set2.json", 'r', encoding='utf-8') as json_file:
        #     self.data2 = [(os.path.join("data/set2", v['filename']), self.trans_label(v['image_labels'])) for k, v in
        #                   json.load(json_file).items()]
        # with open("json/set3.json", 'r', encoding='utf-8') as json_file:
        #     self.data3 = [(os.path.join("data/set3", v['filename']), self.trans_label(v['image_labels']))  for k, v in
        #                   json.load(json_file).items() if self.trans_label(v['image_labels'])!= -1]
        self.all_data = []#self.data3
        self.all_label = []
        a = [0] * 5
        for k in range(5):
            # if k > 0:
            #     continue
            for i in glob.glob(f"data/{data_version}/{k}/*"):
                a[k] += 1
        print(a)

        for k in range(5):
            # if k > 0:
            #     continue
            for i in glob.glob(f"data/{data_version}/{k}/*"):
                if k <= class_type:
                    self.all_data.append(i)
                    self.all_label.append(1)
                else:
                    self.all_data.append(i)
                    self.all_label.append(0)
        # for k in range(5):
        #     for i in glob.glob("data/train7.13/{}/*.jpg".format(k)):
        #         self.all_data.append(i)
        #         self.all_label.append(k)
        # for k in range(5):
        #     for i in glob.glob("data/train7.1/{}/*".format(k)):
        #
        #         self.all_data.append(i)
        #         self.all_label.append(k)
        # indices = list(range(len(self.all_data)))
        # split = int(np.floor(validation_split * len(self.all_data)))
        # np.random.seed(random_seed)
        # np.random.shuffle(indices)
        # train_indices,valid_indices = indices[split:],indices[:split]
        x_train,x_test_val,y_train,y_test_val = train_test_split(self.all_data,self.all_label,test_size = 0.4,
                                                         random_state = random_seed,stratify=self.all_label)
        x_test,x_val,y_test,y_val = train_test_split(x_test_val,y_test_val,test_size = 0.5,
                                                         random_state = random_seed,stratify=y_test_val)
        if train == 0:
            self.data = x_train
            self.label = y_train
        elif train == 1:
            self.data = x_val
            self.label = y_val
        elif train == 2:
            self.data = x_test
            self.label = y_test
        from collections import Counter
        a = [0]*5
        for i in range(5):
            a[i] = self.label.count(i)
        weights = []
        for i in self.label:
            weights.append(1.0/a[i])
        self.weights = weights
        self.c = 2

from torchvision import transforms

    # img_name = glob.glob(f"hist_image/*.jpeg")
    # for i in img_name:
    #     img = Image.open(i)
    #     # img = hisEqulColor(img)
    #     trans = transforms.Compose([
    #         transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
    #     ])
    #     img = trans(img)
    #     img.save(i.replace(".jpeg","") + "_transfer.jpeg")