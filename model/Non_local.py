import torch
from torch import nn
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = x.size(0)
        HH=x.size(2)
        WW=x.size(3)
        

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)


        f = torch.matmul(theta_x, phi_x)

        N = f.size(-1)
        f_div_C = f / N



        # norm_theta_x=torch.norm(theta_x,dim=2,p=2,keepdim=True)
        # norm_phi_x=torch.norm(phi_x,dim=1,p=2,keepdim=True)
        # norm=torch.matmul(norm_theta_x,norm_phi_x)
        # f_div_C=f/(norm+1e-10)


        # norm_theta_x=torch.norm(theta_x,dim=2,p=1,keepdim=True)
        # norm_phi_x=torch.norm(phi_x,dim=1,p=1,keepdim=True)
        # norm=torch.matmul(norm_theta_x,norm_phi_x)
        # f_div_C=f/(norm+1e-10)


        confidence = torch.mean(f_div_C, dim=1).view(batch_size, 1, HH, WW)  # 6*6
        confidence=torch.sigmoid(confidence)
        ret=confidence*x#+x

        return ret



class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch

    for (sub_sample, bn_layer) in [(False, False), (False, False), (True, False), (False, True)]:
        img = Variable(torch.zeros(2, 256, 6, 6))
        net = NONLocalBlock2D(256, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())


        break
