import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology

class PASSRnet(nn.Module):
    def __init__(self, upscale_factor):
        super(PASSRnet, self).__init__()
        ### feature extraction
        self.init_feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            ResB(64),
            ResASPPB(64),
            ResB(64),
            ResASPPB(64),
            ResB(64),
        )
        ### paralax attention
        self.pam = PAM(64)
        ### upscaling
        self.upscale = nn.Sequential(
            ResB(64),
            ResB(64),
            ResB(64),
            ResB(64),
            nn.Conv2d(64, 64 * upscale_factor ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, 3, 1, 1, bias=False),
            nn.Conv2d(3, 3, 3, 1, 1, bias=False)
        )
    def forward(self, x_left, x_right, is_training):
        ### feature extraction
        buffer_left = self.init_feature(x_left)
        buffer_right = self.init_feature(x_right)
        if is_training == 1:
            ### parallax attention
            buffer, (M_right_to_left, M_left_to_right), (M_left_right_left, M_right_left_right), \
            (V_left_to_right, V_right_to_left) = self.pam(buffer_left, buffer_right, is_training)
            ### upscaling
            out = self.upscale(buffer)
            return out, (M_right_to_left, M_left_to_right), (M_left_right_left, M_right_left_right), \
                   (V_left_to_right, V_right_to_left)
        if is_training == 0:
            ### parallax attention
            buffer = self.pam(buffer_left, buffer_right, is_training)
            ### upscaling
            out = self.upscale(buffer)
            return out

class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x

class ResASPPB(nn.Module):
    def __init__(self, channels):
        super(ResASPPB, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.b_1 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
        self.b_2 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
        self.b_3 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv1_1(x))
        buffer_1.append(self.conv2_1(x))
        buffer_1.append(self.conv3_1(x))
        buffer_1 = self.b_1(torch.cat(buffer_1, 1))

        buffer_2 = []
        buffer_2.append(self.conv1_2(buffer_1))
        buffer_2.append(self.conv2_2(buffer_1))
        buffer_2.append(self.conv3_2(buffer_1))
        buffer_2 = self.b_2(torch.cat(buffer_2, 1))

        buffer_3 = []
        buffer_3.append(self.conv1_3(buffer_2))
        buffer_3.append(self.conv2_3(buffer_2))
        buffer_3.append(self.conv3_3(buffer_2))
        buffer_3 = self.b_3(torch.cat(buffer_3, 1))

        return x + buffer_1 + buffer_2 + buffer_3

class PAM(nn.Module):
    def __init__(self, channels):
        super(PAM, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(64)
        self.fusion = nn.Conv2d(channels * 2 + 1, channels, 1, 1, 0, bias=True)
    def __call__(self, x_left, x_right, is_training):
        b, c, h, w = x_left.shape
        buffer_left = self.rb(x_left)
        buffer_right = self.rb(x_right)

        ### M_{right_to_left}
        Q = self.b1(buffer_left).permute(0, 2, 3, 1)                                                # B * H * W * C
        S = self.b2(buffer_right).permute(0, 2, 1, 3)                                               # B * H * C * W
        score = torch.bmm(Q.contiguous().view(-1, w, c),
                          S.contiguous().view(-1, c, w))                                            # (B*H) * W * W
        M_right_to_left = self.softmax(score)

        ### M_{left_to_right}
        Q = self.b1(buffer_right).permute(0, 2, 3, 1)                                               # B * H * W * C
        S = self.b2(buffer_left).permute(0, 2, 1, 3)                                                # B * H * C * W
        score = torch.bmm(Q.contiguous().view(-1, w, c),
                          S.contiguous().view(-1, c, w))                                            # (B*H) * W * W
        M_left_to_right = self.softmax(score)

        ### valid masks
        V_left_to_right = torch.sum(M_left_to_right.detach(), 1) > 0.1
        V_left_to_right = V_left_to_right.view(b, 1, h, w)                                          #  B * 1 * H * W
        V_left_to_right = morphologic_process(V_left_to_right)
        if is_training==1:
            V_right_to_left = torch.sum(M_right_to_left.detach(), 1) > 0.1
            V_right_to_left = V_right_to_left.view(b, 1, h, w)                                      #  B * 1 * H * W
            V_right_to_left = morphologic_process(V_right_to_left)

            M_left_right_left = torch.bmm(M_right_to_left, M_left_to_right)
            M_right_left_right = torch.bmm(M_left_to_right, M_right_to_left)

        ### fusion
        buffer = self.b3(x_right).permute(0,2,3,1).contiguous().view(-1, w, c)                      # (B*H) * W * C
        buffer = torch.bmm(M_right_to_left, buffer).contiguous().view(b, h, w, c).permute(0,3,1,2)  #  B * C * H * W
        out = self.fusion(torch.cat((buffer, x_left, V_left_to_right), 1))

        ## output
        if is_training == 1:
            return out, \
               (M_right_to_left.contiguous().view(b, h, w, w), M_left_to_right.contiguous().view(b, h, w, w)), \
               (M_left_right_left.view(b,h,w,w), M_right_left_right.view(b,h,w,w)), \
               (V_left_to_right, V_right_to_left)
        if is_training == 0:
            return out

def morphologic_process(mask):
    device = mask.device
    b,_,_,_ = mask.shape
    mask = 1-mask
    mask_np = mask.cpu().numpy().astype(bool)
    mask_np = morphology.remove_small_objects(mask_np, 20, 2)
    mask_np = morphology.remove_small_holes(mask_np, 10, 2)
    for idx in range(b):
        buffer = np.pad(mask_np[idx,0,:,:],((3,3),(3,3)),'constant')
        buffer = morphology.binary_closing(buffer, morphology.disk(3))
        mask_np[idx,0,:,:] = buffer[3:-3,3:-3]
    mask_np = 1-mask_np
    mask_np = mask_np.astype(float)

    return torch.from_numpy(mask_np).float().to(device)