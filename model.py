
import numpy as np
import torch
import torch.nn as nn

img_shape = (1,128,128)

class Generator(nn.Module):
    def __init__(self,input_dim):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(input_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),#np.prod(img_shape)连乘操作，长*宽*深度
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),#改进1、判别器最后一层去掉sigmoid。sigmoid函数容易出现梯度消失的情况。
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity






class lstm(nn.Module):
    def __init__(self, input_size = 16384, hidden_size = 512, output_size = 100, num_layer=2):
        super(lstm, self).__init__()
        self.layer1 = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer,
                                   bias=True, batch_first=False, dropout=0.5, bidirectional=False)
        # self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x): #x[b = 1 ,4,16384]
        x = x.permute(1,0,2)
        x, _ = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        x = x.permute(1,0,2)

        return x

ls = lstm()

a = torch.randn([2,4,16384])

b = ls(a)
