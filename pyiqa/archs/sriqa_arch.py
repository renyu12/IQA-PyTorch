r"""SRIQA Metric

# renyu: 使用BSRGAN预训练模型作为Backbone
"""

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from pyiqa.utils.registry import ARCH_REGISTRY
#from pyiqa.archs.arch_util import load_pretrained_network    # renyu: 仅使用自定义部分加载预训练模型方法
from pyiqa.archs.arch_util import random_crop, uniform_crop   # renyu: 做多次crop测试评分


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

@ARCH_REGISTRY.register()
class SRIQA(nn.Module):
    def __init__(
        self,
        metric_type='NR',
        model_name='sriqa_nr',
        weighted_average=True,
        pretrained_model_path=None,
        load_feature_weight_only=True,
        crop_num=15
    ):
        super(SRIQA, self).__init__()
        self.crop_num = crop_num

        RRDB_block_f = functools.partial(RRDB, nf=64, gc=32)

        self.conv_first = nn.Conv2d(3, 64, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, 23)
        self.trunk_conv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        
        # renyu: 增加一层后卷积
        self.trunk_conv2 = nn.Sequential(
            nn.ReLU(),    # renyu: 这是给前面trunk_conv层的
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.ReLU()
        )

        # renyu: upsampling部分全部移除，改为MLP，输入是224x224x64，想办法设计合适的回归Head处理
        # renyu: 方案1 直接全局平均池化然后64维FC输出
        #self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化，直接忽略空间信息224*224变成1  
        #self.fc = nn.Linear(64, 1)     # 全连接层，输出最后回归值
        # renyu: 方案2 平均池化到2*2，然后MLP 256->128->1
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))  # 全局平均池化到2*2
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        # renyu: 方案3 最大池化到4*4 然后maniqa MLP Head
        #self.global_pool = nn.AdaptiveMaxPool2d((4, 4))
        #self.fc = nn.Sequential(
        #    nn.Linear(4 * 4 * 64, 128),
        #    nn.ReLU(),
        #    nn.Dropout(0.1),
        #    nn.Linear(128, 1),
        #    nn.ReLU()
        #)
        # renyu: 方案4 maniqa双路 MLP Head
        #self.global_pool = nn.AdaptiveAvgPool2d((2, 2))  # 全局平均池化到2*2
        #self.fc = nn.Sequential(
        #    nn.Linear(256, 128),
        #    nn.ReLU(),
        #    nn.Dropout(0.1),
        #    nn.Linear(128, 1),
        #    nn.ReLU()
        #)
        #self.fc_weight = nn.Sequential(
        #    nn.Linear(256, 128),
        #    nn.ReLU(),
        #    nn.Dropout(0.1),
        #    nn.Linear(128, 1),
        #    nn.Sigmoid()
        #)

        # renyu: 最后处理冻结Backbone，不然可能一些层没定义好load不进来
        #freeze_layers = (self.conv_first, self.RRDB_trunk, self.trunk_conv)            # renyu: 最后的卷积层解冻后发现效果比较好
        freeze_layers = (self.conv_first, self.RRDB_trunk)
        #freeze_layers = (self.conv_first, self.RRDB_trunk[0:22])    # renyu: 共23层RRDB，多解冻一个试试

        if pretrained_model_path is not None:
            self.load_pretrained_network(pretrained_model_path, load_feature_weight_only)

            # renyu: BSRGAN指定层参数全部冻结
            for freeze_layer in freeze_layers:
                for layer_params in freeze_layer.parameters():
                    layer_params.requires_grad = False


    # renyu: 加载预训练模型，如果load_feature_weight_only=True说明是加载的BSRGAN参数，否则就是加载完整SRIQA模型
    def load_pretrained_network(self, model_path, load_feature_weight_only):
        print(f'Loading pretrained model from {model_path}')
        #state_dict = torch.load(model_path, map_location=torch.device('cpu'))['params']
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # renyu: 加载BSRGAN Backbone参数，回归头从头训练
        if load_feature_weight_only:
            print('Only load backbone feature net')
            new_state_dict = {}
            # renyu: TODO: 这里过滤下BSRGAN中需要的层
            for k in state_dict.keys():
                if 'RRDB_trunk' in k or 'conv_first' in k or 'trunk_conv' in k:
                    new_state_dict[k] = state_dict[k]
            self.load_state_dict(new_state_dict, strict=False)
        
        # renyu: 加载训练好的SRIQA模型继续训练
        else:
            state_dict = state_dict['params']    # renyu: IQA-Pytorch框架保存的模型多一层params字典，硬编码解析下
            self.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        # renyu: 预处理随机crop，训练阶段不多次crop，
        bsz = x.shape[0]    # renyu: B C H W
        if self.training:
            x = random_crop(x, crop_size=224, crop_num=1)
        else:
            x = uniform_crop(x, crop_size=224, crop_num=self.crop_num)            # renyu: B*Crop C H W

        # renyu: x输入设定为224x224x3通道
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        if hasattr(self, 'trunk_conv2'):
            fea = self.trunk_conv2(fea)
        print(fea.shape)    # renyu: 方便看下进度
        
        # renyu: 拿到feature 224x224x64通道后，直接过回归Head
        out = self.global_pool(fea)                     # 池化后的尺寸为 (batch_size, 64, 1, 1)  
        out = out.view(out.size(0), -1)                 # 扁平化，尺寸为 (batch_size, 64)  

        # renyu: 双路MLP带权重做加权处理，否则直接出结果
        if hasattr(self, 'fc_weight'):
            per_patch_score = self.fc(out)    # renyu: B*Crop 1
            per_patch_score = per_patch_score.reshape(bsz, -1)    # renyu: B Crop
            per_patch_weight = self.fc_weight(out)
            per_patch_weight = per_patch_weight.reshape(bsz, -1)

            score = (per_patch_weight * per_patch_score).sum(dim=-1) / (per_patch_weight.sum(dim=-1) + 1e-8)
            out = score.unsqueeze(1)
        # renyu: 开了多crop未做双路MLP加权求和，就是直接取平均（即使当前实际crop_num=1也兼容）
        elif self.crop_num > 1:
            per_patch_score = self.fc(out)    # renyu: B*Crop 1
            per_patch_score = per_patch_score.reshape(bsz, -1)    # renyu: B Crop
            out = per_patch_score.sum(dim=-1, keepdim=True)  
        else:
            out = self.fc(out)                              # 应用全连接层  

        return out

# renyu: 原版的BSRGAN，不需要了
'''
@ARCH_REGISTRY.register()
class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.sf = sf
        print([in_nc, out_nc, nf, nb, gc, sf])

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.sf==4:
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.sf==4:
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
'''
