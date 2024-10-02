# renyu: 预提取特征测试不同的回归头

import torch
import torch.nn as nn
import torch.nn.functional as F
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import load_pretrained_network

import timm    # renyu: 用timm库引入ResNet50模型


default_model_urls = {
    'koniq10k': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/CNNIQA_koniq10k-e6f14c91.pth'
}


@ARCH_REGISTRY.register()
class LRHead(nn.Module): 
    def __init__(self, repeat_crop=False, crop_num=1, pretrained_model_path=None):  
        super(LRHead, self).__init__()  
        # renyu: 后卷积
        self.head_conv = nn.Sequential(
            nn.ReLU(),    # renyu: 这是给前面trunk_conv层的
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.ReLU()
        )

        # renyu: 方案1 直接全局平均池化然后64维FC输出
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化，直接忽略空间信息224*224变成1  
        self.fc = nn.Linear(64, 1)     # 全连接层，输出最后回归值

    def forward(self, x):
        # renyu: x输入设定为224x224x3通道
        fea = self.head_conv(x)
        print(fea.shape)    # renyu: 方便看下进度
        
        # renyu: 拿到feature 224x224x64通道后，直接过回归Head
        out = self.global_pool(fea)                     # 池化后的尺寸为 (batch_size, 64, 1, 1)  
        out = out.view(out.size(0), -1)                 # 扁平化，尺寸为 (batch_size, 64)  
        out = self.fc(out)                              # 应用全连接层  

        return out

@ARCH_REGISTRY.register()
class AVGMLPHead(nn.Module): 
    def __init__(self, repeat_crop=False, crop_num=1, pretrained_model_path=None):  
        super(AVGMLPHead, self).__init__()  
        self.repeat_crop = repeat_crop
        self.crop_num = crop_num

        # renyu: 后卷积
        self.head_conv = nn.Sequential(
            nn.ReLU(),    # renyu: 这是给前面trunk_conv层的
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.ReLU()
        )

        # renyu: 方案2 平均池化到2*2，然后MLP 256->128->1
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))  # 全局平均池化到2*2
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

        # renyu: 要冻结的层这里设置
        #freeze_layers = (self.conv_first, self.RRDB_trunk)
        freeze_layers = []

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
        
        state_dict = state_dict['params']    # renyu: IQA-Pytorch框架保存的模型多一层params字典，硬编码解析下
        self.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        # renyu: 预处理随机crop，训练阶段不多次crop，
        bsz = x.shape[0]    # renyu: B C H W
        if self.repeat_crop:
            if self.training:
                x = random_crop(x, crop_size=224, crop_num=1)
            else:
                x = uniform_crop(x, crop_size=224, crop_num=self.crop_num)            # renyu: B*Crop C H W

        # renyu: x输入设定为224x224x3通道
        fea = self.head_conv(x)
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
            real_crop_num = 1    # renyu: 兼容训练阶段和测试阶段crop_num不一致，直接计算一下
            if out.shape[0] != bsz:
                real_crop_num = out.shape[0] / bsz
            per_patch_score = per_patch_score.reshape(bsz, -1)    # renyu: B Crop
            out = per_patch_score.sum(dim=-1, keepdim=True) / real_crop_num
        else:
            out = self.fc(out)                              # 应用全连接层  

        return out


@ARCH_REGISTRY.register()
class MAXMLPHead(nn.Module): 
    def __init__(self, repeat_crop=False, crop_num=1, pretrained_model_path=None):  
        super(MAXMLPHead, self).__init__()  
        self.repeat_crop = repeat_crop
        self.crop_num = crop_num

        # renyu: 后卷积
        self.head_conv = nn.Sequential(
            nn.ReLU(),    # renyu: 这是给前面trunk_conv层的
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.ReLU()
        )

        # renyu: 方案3 最大池化到4*4 然后maniqa MLP Head
        self.global_pool = nn.AdaptiveMaxPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.ReLU()
        )

        # renyu: 要冻结的层这里设置
        #freeze_layers = (self.conv_first, self.RRDB_trunk)
        freeze_layers = []

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
        
        state_dict = state_dict['params']    # renyu: IQA-Pytorch框架保存的模型多一层params字典，硬编码解析下
        self.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        # renyu: 预处理随机crop，训练阶段不多次crop，
        bsz = x.shape[0]    # renyu: B C H W
        if self.repeat_crop:
            if self.training:
                x = random_crop(x, crop_size=224, crop_num=1)
            else:
                x = uniform_crop(x, crop_size=224, crop_num=self.crop_num)            # renyu: B*Crop C H W

        # renyu: x输入设定为224x224x3通道
        fea = self.head_conv(x)
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
            real_crop_num = 1    # renyu: 兼容训练阶段和测试阶段crop_num不一致，直接计算一下
            if out.shape[0] != bsz:
                real_crop_num = out.shape[0] / bsz
            per_patch_score = per_patch_score.reshape(bsz, -1)    # renyu: B Crop
            out = per_patch_score.sum(dim=-1, keepdim=True) / real_crop_num
        else:
            out = self.fc(out)                              # 应用全连接层  

        return out

@ARCH_REGISTRY.register()
class DoubleMLPHead(nn.Module): 
    def __init__(self, repeat_crop=False, crop_num=1, pretrained_model_path=None):  
        super(DoubleMLPHead, self).__init__()  
        self.repeat_crop = repeat_crop
        self.crop_num = crop_num

        # renyu: 后卷积
        self.head_conv = nn.Sequential(
            nn.ReLU(),    # renyu: 这是给前面trunk_conv层的
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.ReLU()
        )

        # renyu: 方案4 maniqa双路 MLP Head
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))  # 全局平均池化到2*2
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # renyu: 要冻结的层这里设置
        #freeze_layers = (self.conv_first, self.RRDB_trunk)
        freeze_layers = []

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
        
        state_dict = state_dict['params']    # renyu: IQA-Pytorch框架保存的模型多一层params字典，硬编码解析下
        self.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        # renyu: 预处理随机crop，训练阶段不多次crop，
        bsz = x.shape[0]    # renyu: B C H W
        if self.repeat_crop:
            if self.training:
                x = random_crop(x, crop_size=224, crop_num=1)
            else:
                x = uniform_crop(x, crop_size=224, crop_num=self.crop_num)            # renyu: B*Crop C H W

        # renyu: x输入设定为224x224x3通道
        fea = self.head_conv(x)
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
            real_crop_num = 1    # renyu: 兼容训练阶段和测试阶段crop_num不一致，直接计算一下
            if out.shape[0] != bsz:
                real_crop_num = out.shape[0] / bsz
            per_patch_score = per_patch_score.reshape(bsz, -1)    # renyu: B Crop
            out = per_patch_score.sum(dim=-1, keepdim=True) / real_crop_num
        else:
            out = self.fc(out)                              # 应用全连接层  

        return out

@ARCH_REGISTRY.register()
class SimpleCNNIQA(nn.Module): 
    def __init__(self, repeat_crop=False, crop_num=1, pretrained_model_path=None):  
        super(SimpleCNNIQA, self).__init__()  
        self.repeat_crop = repeat_crop
        self.crop_num = crop_num

        # renyu: 后卷积
        self.head_conv = nn.Sequential(
            nn.ReLU(),    # renyu: 这是给前面trunk_conv层的
            nn.Conv2d(3, 64, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.ReLU()
        )

        # renyu: 方案2 平均池化到2*2，然后MLP 256->128->1
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))  # 全局平均池化到2*2
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

        # renyu: 要冻结的层这里设置
        #freeze_layers = (self.conv_first, self.RRDB_trunk)
        freeze_layers = []

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
        
        state_dict = state_dict['params']    # renyu: IQA-Pytorch框架保存的模型多一层params字典，硬编码解析下
        self.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        # renyu: 预处理随机crop，训练阶段不多次crop，
        bsz = x.shape[0]    # renyu: B C H W
        if self.repeat_crop:
            if self.training:
                x = random_crop(x, crop_size=224, crop_num=1)
            else:
                x = uniform_crop(x, crop_size=224, crop_num=self.crop_num)            # renyu: B*Crop C H W

        # renyu: x输入设定为224x224x3通道
        fea = self.head_conv(x)
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
            real_crop_num = 1    # renyu: 兼容训练阶段和测试阶段crop_num不一致，直接计算一下
            if out.shape[0] != bsz:
                real_crop_num = out.shape[0] / bsz
            per_patch_score = per_patch_score.reshape(bsz, -1)    # renyu: B Crop
            out = per_patch_score.sum(dim=-1, keepdim=True) / real_crop_num
        else:
            out = self.fc(out)                              # 应用全连接层  

        return out