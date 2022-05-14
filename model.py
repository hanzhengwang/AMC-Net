import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
from torch.autograd import Variable


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)]
        feat_block += [nn.BatchNorm1d(low_dim)]

        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block

    def forward(self, x):
        x = self.feat_block(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]

        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        return x

    # Define the ResNet18-based Model


class visible_net_resnet(nn.Module):
    def __init__(self, arch='resnet18'):
        super(visible_net_resnet, self).__init__()
        if arch == 'resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch == 'resnet50':
            model_ft = models.resnet50(pretrained=True)

        for mo in model_ft.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.visible = model_ft
        self.dropout = nn.Dropout(p=0.5)
        self.new1 = CAB(256)
        self.new2 = CAB(512)

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        x = self.visible.layer1(x)
        new1, _ = self.new1(x)
        x = self.visible.layer2(new1)
        new2, _ = self.new2(x)
        x = self.visible.layer3(new2)


        return x

class thermal_net_resnet(nn.Module):
    def __init__(self, arch='resnet18'):
        super(thermal_net_resnet, self).__init__()
        if arch == 'resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch == 'resnet50':
            model_ft = models.resnet50(pretrained=True)

        for mo in model_ft.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.thermal = model_ft
        self.dropout = nn.Dropout(p=0.5)
        self.new1 = CAB(256)
        self.new2 = CAB(512)

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x = self.thermal.layer1(x)
        new1, _ = self.new1(x)
        x = self.thermal.layer2(new1)
        new2, _ = self.new2(x)
        x = self.thermal.layer3(new2)


        return x
class share_net(nn.Module):
    def __init__(self, arch ='resnet50'):
        super(share_net, self).__init__()
        if arch =='resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch =='resnet50':
            model_ft = models.resnet50(pretrained=True)

        for mo in model_ft.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.share = model_ft
        self.dropout = nn.Dropout(p=0.5)
        self.new = CAB(2048)
    def forward(self, x):
        x = self.share.layer4(x)
        x, mask = self.new(x)
        num_part = 6 # number of part
        # pool size
        sx = x.size(2) / num_part
        sx = int(sx)
        kx = x.size(2) - sx * (num_part-1)
        kx = int(kx)
        x = nn.functional.avg_pool2d(x, kernel_size=(kx, x.size(3)), stride=(sx, x.size(3)))
        #x = self.thermal.avgpool(x)
        x = x.view(x.size(0), x.size(1), x.size(2))
        # x = self.dropout(x)
        return x, mask

class embed_net(nn.Module):
    def __init__(self, low_dim, class_num, drop=0.5, arch='resnet50'):
        super(embed_net, self).__init__()
        if arch == 'resnet18':
            self.visible_net = visible_net_resnet(arch=arch)
            self.thermal_net = thermal_net_resnet(arch=arch)
            pool_dim = 512
        elif arch == 'resnet50':
            self.visible_net = visible_net_resnet(arch=arch)
            self.thermal_net = thermal_net_resnet(arch=arch)
            pool_dim = 2048

        self.feature1 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature2 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature3 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature4 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature5 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature6 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.classifier1 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier2 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier3 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier4 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier5 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier6 = ClassBlock(low_dim, class_num, dropout=drop)
        self.dist = nn.MSELoss(reduction='sum')
        self.l2norm = Normalize(2)
        self.share_net = share_net(arch=arch)

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_net(x1)
            x1, v_mask = self.share_net(x1)
            x2 = self.thermal_net(x2)
            x2, t_mask = self.share_net(x2)
            x1 = x1.chunk(6, 2)
            x1_0 = x1[0].contiguous().view(x1[0].size(0), -1)
            x1_1 = x1[1].contiguous().view(x1[1].size(0), -1)
            x1_2 = x1[2].contiguous().view(x1[2].size(0), -1)
            x1_3 = x1[3].contiguous().view(x1[3].size(0), -1)
            x1_4 = x1[4].contiguous().view(x1[4].size(0), -1)
            x1_5 = x1[5].contiguous().view(x1[5].size(0), -1)

            x2 = x2.chunk(6, 2)
            x2_0 = x2[0].contiguous().view(x2[0].size(0), -1)
            x2_1 = x2[1].contiguous().view(x2[1].size(0), -1)
            x2_2 = x2[2].contiguous().view(x2[2].size(0), -1)
            x2_3 = x2[3].contiguous().view(x2[3].size(0), -1)
            x2_4 = x2[4].contiguous().view(x2[4].size(0), -1)
            x2_5 = x2[5].contiguous().view(x2[5].size(0), -1)
            x_0 = torch.cat((x1_0, x2_0), 0)
            x_1 = torch.cat((x1_1, x2_1), 0)
            x_2 = torch.cat((x1_2, x2_2), 0)
            x_3 = torch.cat((x1_3, x2_3), 0)
            x_4 = torch.cat((x1_4, x2_4), 0)
            x_5 = torch.cat((x1_5, x2_5), 0)
            loss = self.dist(v_mask, t_mask)
        elif modal == 1:
            x= self.visible_net(x1)
            x, _ = self.share_net(x)
            x = x.chunk(6, 2)
            x_0 = x[0].contiguous().view(x[0].size(0), -1)
            x_1 = x[1].contiguous().view(x[1].size(0), -1)
            x_2 = x[2].contiguous().view(x[2].size(0), -1)
            x_3 = x[3].contiguous().view(x[3].size(0), -1)
            x_4 = x[4].contiguous().view(x[4].size(0), -1)
            x_5 = x[5].contiguous().view(x[5].size(0), -1)
        elif modal == 2:
            x = self.thermal_net(x2)
            x, _ = self.share_net(x)
            x = x.chunk(6, 2)
            x_0 = x[0].contiguous().view(x[0].size(0), -1)
            x_1 = x[1].contiguous().view(x[1].size(0), -1)
            x_2 = x[2].contiguous().view(x[2].size(0), -1)
            x_3 = x[3].contiguous().view(x[3].size(0), -1)
            x_4 = x[4].contiguous().view(x[4].size(0), -1)
            x_5 = x[5].contiguous().view(x[5].size(0), -1)

        y_0 = self.feature1(x_0)
        y_1 = self.feature2(x_1)
        y_2 = self.feature3(x_2)
        y_3 = self.feature4(x_3)
        y_4 = self.feature5(x_4)
        y_5 = self.feature6(x_5)
        # y = self.feature(x)
        out_0 = self.classifier1(y_0)
        out_1 = self.classifier2(y_1)
        out_2 = self.classifier3(y_2)
        out_3 = self.classifier4(y_3)
        out_4 = self.classifier5(y_4)
        out_5 = self.classifier6(y_5)
        # out = self.classifier(y)
        if self.training:
            return (out_0, out_1, out_2, out_3, out_4, out_5), (
            self.l2norm(y_0), self.l2norm(y_1), self.l2norm(y_2), self.l2norm(y_3), self.l2norm(y_4), self.l2norm(y_5)), loss
        else:
            x_0 = self.l2norm(x_0)
            x_1 = self.l2norm(x_1)
            x_2 = self.l2norm(x_2)
            x_3 = self.l2norm(x_3)
            x_4 = self.l2norm(x_4)
            x_5 = self.l2norm(x_5)
            x = torch.cat((x_0, x_1, x_2, x_3, x_4, x_5), 1)
            y_0 = self.l2norm(y_0)
            y_1 = self.l2norm(y_1)
            y_2 = self.l2norm(y_2)
            y_3 = self.l2norm(y_3)
            y_4 = self.l2norm(y_4)
            y_5 = self.l2norm(y_5)
            y = torch.cat((y_0, y_1, y_2, y_3, y_4, y_5), 1)
            return x, y

class CAB(nn.Module):
    expansion = 4
    # 357
    def __init__(self, inplanes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d, k_size=3):
        super(CAB, self).__init__()
        planes = inplanes // 2
        outplanes = planes // 2
        self.conv11 = nn.Conv2d(inplanes, planes, 3, stride,
                               padding=3, dilation=3, bias=False)
        self.bn11 = norm_layer(planes)

        self.conv12 = nn.Conv2d(inplanes, planes, 3, stride,
                               padding=5, dilation=5, bias=False)
        self.bn12 = norm_layer(planes)

        self.conv13 = nn.Conv2d(inplanes, planes, 3, stride,
                               padding=7, dilation=7, bias=False)
        self.bn13 = norm_layer(planes)

        self.conv21 = nn.Conv2d(planes, outplanes, 1, stride,
                                1, dilation=1, bias=False)
        self.bn21 = norm_layer(outplanes)

        self.conv22 = nn.Conv2d(planes, outplanes, 1, stride,
                                1, dilation=1, bias=False)
        self.bn22 = norm_layer(outplanes)

        self.conv23 = nn.Conv2d(planes, outplanes, 1, stride,
                                1, dilation=1, bias=False)
        self.bn23 = norm_layer(outplanes)

        self.conv = nn.Conv2d(outplanes, 1, 1, stride, 1, dilation=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        #self.ACB = ACB()
        self.CONV1 = nn.Conv2d(inplanes, planes, 3, stride, 1, dilation=1, bias=False)
        self.CONV2 = nn.Conv2d(planes, outplanes, 1, stride, 1, dilation=1, bias=False)
        self.BN1 = norm_layer(planes)
        self.BN2 = norm_layer(outplanes)

    def forward(self, x):
        size = x.size()[2:]
        identity = x
        #-----------------#
        OUT1 = self.BN1(self.CONV1(x))
        OUT2 = self.BN2(self.CONV2(OUT1))

        out11 = self.bn11(self.conv11(x))
        out12 = self.bn12(self.conv12(x))
        out13 = self.bn13(self.conv13(x))

        out21 = self.bn21(self.conv21(out11))
        out22 = self.bn22(self.conv22(out12))
        out23 = self.bn23(self.conv23(out13))

        out21 = F.interpolate(out21, size, mode='bilinear', align_corners=True)
        out22 = F.interpolate(out22, size, mode='bilinear', align_corners=True)
        out23 = F.interpolate(out23, size, mode='bilinear', align_corners=True)
        OUT2 = F.interpolate(OUT2, size, mode='bilinear', align_corners=True)

        out = out21 + out22 + out23 + OUT2

        out = self.conv(out)
        out = self.sigmoid(out)

        out = F.interpolate(out, size, mode='bilinear', align_corners=True)

        out_pam = x * out
        #-----------------#
        output = out_pam + identity

        return output, out

class ACB(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(ACB, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        #------------------------------------#
        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)
        y = y1 + y2
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)
        return x * y.expand_as(x)

