import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import resnet
# import bit_model
from wide_resnet import WideResNet


import copy

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50, Bottleneck

class EyeNet(nn.Module):
    def __init__(self,num_classes,pretrained=True,model_name = "resnet101"):
        super(EyeNet, self).__init__()
        self.num_classes = num_classes
        self.backbone = None
        self.model_name = model_name


        if model_name == "resnet18":
            self.backbone = resnet.resnet18(pretrained=pretrained, num_classes=self.num_classes)
        elif model_name == "resnet34":

            self.backbone = resnet.resnet34(pretrained=pretrained, num_classes=self.num_classes)
        elif model_name == "resnet101":
            self.backbone = resnet.resnet101(pretrained=pretrained, num_classes=self.num_classes)
        elif model_name == "resnet50":
            self.backbone = resnet.resnet50(pretrained=pretrained,num_classes= self.num_classes)
        '''
        elif model_name == "dla":
            self.backbone = dla.dla60x(pretrained="imagenet",num_classes=num_classes)
        elif model_name == "efficient":
            self.backbone= EfficientNet.from_pretrained('efficientnet-b7',num_classes=num_classes)
        # elif model_name == "bit":
        #     self.backbone = bit_model.KNOWN_MODELS["BiT-M-R101x1"](head_size=num_classes, zero_head=True)
        #     self.backbone.load_from(np.load("BiT-M-R101x1.npz"))
        elif model_name == "mgn":
            self.backbone = MGN(num_classes=num_classes)
        elif model_name == "wide_resnet":
            # self.backbone = WideResNet(depth=28,num_classes=5,widen_factor=4,dropRate=0.4)
            self.backbone = resnet.wide_resnet50_2(pretrained=True,num_classes=num_classes)
        elif model_name == "simple":
            pass
        '''
            # self.backbone =
         # = ###mobileNet.mobilenet_v2(True)
        # self.se_fc1 = nn.Linear(2048, 128)
        # self.se_fc2 = nn.Linear(128, 2048)
        # self.bn = nn.BatchNorm2d(2048)
        self.bn1 = nn.BatchNorm1d(512*self.backbone.block_expansion)
        self.relu = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(0.25)
        self.linear1 = nn.Linear(512*self.backbone.block_expansion, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, num_classes)

        self.myfc = nn.Linear(512*self.backbone.block_expansion, num_classes)
        # self.myfc = ArcMarginModel(512*self.backbone.block_expansion,num_classes)
        # self.FPA = FPA()
    def forward(self,x,target=None):
        x = self.backbone(x)
        x = self.bn1(x)
        x = self.drop1(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.linear2(x)

        # x = self.FPA(x)
        # x_ori = x
        # x = nn.AdaptiveAvgPool2d((1, 1))(x)
        # x = torch.flatten(x, 1)
        # x = self.se_fc1(x)
        # x = self.se_fc2(x)
        # x = nn.Sigmoid()(x)
        # x = x_ori * x
        # x = nn.AdaptiveAvgPool2d((1, 1))(x)
        # x = torch.flatten(x, 1)
        # x = self.bn(x)
        # x = self.myfc(x)
        return x

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig
if __name__ == "__main__":
    x = EyeNet(5,pretrained=False,model_name = "bit")
    inter_feature = {}


    # def make_hook(name):
    #     def hook(m, input, output):
    #         inter_feature[name] = input
    #     return hook


    # def hook(m, input, output):
    #     inter_feature["feature"] = input
    # # x.register_forward_hook(make_hook("backbone.avgpool"))
    # for k, v in x.named_children():
    #     for z, q in v.named_children():
    #        if z == "newfc":
    #             q.register_forward_hook(hook)
    # img = torch.rand((10,3,224,224))
    # y = x(img)
    # data = []
    # for emb in inter_feature["feature"][0]:
    #     single_data = emb.detach().numpy().reshape(-1)
    #     data.append(single_data)
    # tsne = TSNE(n_components=2, init='pca', random_state=0)
    #
    # label = np.random.randint(5,size=10)
    # result = tsne.fit_transform(np.array(data))
    # fig = plot_embedding(result, label,
    #                      't-SNE embedding of the digits (time %.2fs)'
    #                      % (99))
    #
    # plt.savefig("good.jpg")
    # print(inter_feature["feature"][0][0].shape)
