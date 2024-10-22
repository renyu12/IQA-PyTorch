r"""CNNIQA Model.

Zheng, Heliang, Huan Yang, Jianlong Fu, Zheng-Jun Zha, and Jiebo Luo.
"Learning conditional knowledge distillation for degraded-reference image
quality assessment." In Proceedings of the IEEE/CVF International Conference
on Computer Vision (ICCV), pp. 10242-10251. 2021.

Ref url: https://github.com/lidq92/CNNIQA
Re-implemented by: Chaofeng Chen (https://github.com/chaofengc) with modification:
    - We use 3 channel RGB input.

"""

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
class CNNIQA(nn.Module):
    r"""CNNIQA model.
    Args:
        - ker_size (int): Kernel size.
        - n_kers (int): Number of kernals.
        - n1_nodes (int): Number of n1 nodes.
        - n2_nodes (int): Number of n2 nodes.
        - pretrained_model_path (String): Pretrained model path.

    """

    def __init__(
        self,
        ker_size=7,
        n_kers=50,
        n1_nodes=800,
        n2_nodes=800,
        pretrained='koniq10k',
        pretrained_model_path=None,
    ):
        super(CNNIQA, self).__init__()

        self.conv1 = nn.Conv2d(3, n_kers, ker_size)
        self.fc1 = nn.Linear(2 * n_kers, n1_nodes)
        self.fc2 = nn.Linear(n1_nodes, n2_nodes)
        self.fc3 = nn.Linear(n2_nodes, 1)
        self.dropout = nn.Dropout()

        if pretrained_model_path is None and pretrained is not None:
            pretrained_model_path = default_model_urls[pretrained]

        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True, 'params')

    def forward(self, x):
        r"""Compute IQA using CNNIQA model.

        Args:
            x: An input tensor with (N, C, H, W) shape. RGB channel order for colour images.

        Returns:
            Value of CNNIQA model.

        """
        h = self.conv1(x)

        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h = torch.cat((h1, h2), 1)  # max-min pooling
        h = h.squeeze(3).squeeze(2)

        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))

        q = self.fc3(h)
        return q

@ARCH_REGISTRY.register()
class ResNetIQA(nn.Module):  
    def __init__(self, freeze_resnet=True, pretrained_model_path=None):  
        super(ResNetIQA, self).__init__()  
        # 创建预训练的 ResNet-50 模型  
        self.resnet50 = timm.create_model('resnet50', pretrained=True)  
        # 冻结 ResNet-50 的参数  
        if freeze_resnet:  
            for param in self.resnet50.parameters():  
                param.requires_grad = False  
            #for param in self.resnet50.layer4.parameters():    # renyu: 解冻layer4效果更好
            #    param.requires_grad = True
          
        # 移除 ResNet-50 的分类头（最后的全连接层）  
        self.resnet50.fc = nn.Identity()  # 或者使用 nn.Sequential() 空操作以保持兼容性  
  
        # 添加两层 MLP 回归头  
        self.mlp = nn.Sequential(  
            nn.Linear(2048, 512),  # 假设 ResNet-50 的特征维度是 2048  
            nn.ReLU(),  
            nn.Dropout(0.1),
            nn.Linear(512, 1)      # 回归输出一个值  
        )  

        # renyu: 支持一下加载预训练模型，直接给路径即可
        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True, 'params')
      
    def forward(self, x):  
        # 通过 ResNet-50 提取特征  
        features = self.resnet50(x)  
        # 通过 MLP 回归头得到预测值  
        output = self.mlp(features)  
        return output  

# renyu: 尝试取ResNet分层特征的代码（TODO: 待调试）
@ARCH_REGISTRY.register()
class ResNetIQABackbone(nn.Module):
    def __init__(
        self,
        network='resnet50',
        test_sample=10,
        pretrained_model_path=None,
    ):
        super().__init__()

        self.test_sample = test_sample

        if network == 'resnet50':
            self.model = models.resnet50(weights='IMAGENET1K_V1')
        elif network == 'resnet34':
            self.model = models.resnet34(weights='IMAGENET1K_V1')
        elif network == 'resnet18':
            self.model = models.resnet18(weights='IMAGENET1K_V1')



    def forward_backbone(self, model, x):
        # See note [TorchScript super()]
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        l1 = x
        x = model.layer2(x)
        l2 = x
        x = model.layer3(x)
        l3 = x
        x = model.layer4(x)
        l4 = x
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = model.fc(x)

        return x, l1, l2, l3, l4

    def forward(self, x):
        #x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        bsz = x.shape[0]

        '''
        if self.training:
            x = random_crop(x, 224, 1)
            num_patches = 1
        else:
            x = uniform_crop(x, 224, self.test_sample)
            num_patches = self.test_sample
        '''

        out, layer1, layer2, layer3, layer4 = self.forward_backbone(self.model, x)
        return layer4