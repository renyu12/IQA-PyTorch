import torch

from collections import OrderedDict
from pyiqa.default_model_configs import DEFAULT_CONFIGS
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.utils.img_util import imread2tensor

from pyiqa.losses.loss_util import weight_reduce_loss
from pyiqa.archs.arch_util import load_pretrained_network

# renyu: 命令行启动就是直接创建推理模型走到这里，非常简单创建一个默认参数的模型，不太支持自定义配置
#        根据模型的不同会读取default_model_configs.py文件中的一点自定义推理配置
#        其他配置就是直接按model的默认参数跑
class InferenceModel(torch.nn.Module):
    """Common interface for quality inference of images with default setting of each metric."""

    def __init__(
            self,
            metric_name,
            as_loss=False,
            loss_weight=None,
            loss_reduction='mean',
            device=None,
            seed=123,
            check_input=True,
            **kwargs  # Other metric options
    ):
        super(InferenceModel, self).__init__()

        self.metric_name = metric_name

        # ============ set metric properties ===========
        self.lower_better = DEFAULT_CONFIGS[metric_name].get('lower_better', False)
        self.metric_mode = DEFAULT_CONFIGS[metric_name].get('metric_mode', None)
        self.score_range = DEFAULT_CONFIGS[metric_name].get('score_range', None)
        if self.metric_mode is None:
            self.metric_mode = kwargs.pop('metric_mode')
        elif 'metric_mode' in kwargs:
            kwargs.pop('metric_mode')
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.as_loss = as_loss
        self.loss_weight = loss_weight
        self.loss_reduction = loss_reduction
        self.check_input = check_input

        # =========== define metric model ===============
        net_opts = OrderedDict()
        # load default setting first
        if metric_name in DEFAULT_CONFIGS.keys():
            default_opt = DEFAULT_CONFIGS[metric_name]['metric_opts']
            net_opts.update(default_opt)
        # then update with custom setting
        net_opts.update(kwargs)
        network_type = net_opts.pop('type')
        self.net = ARCH_REGISTRY.get(network_type)(**net_opts)
        self.net = self.net.to(self.device)
        self.net.eval()

        self.seed = seed

        self.dummy_param = torch.nn.Parameter(torch.empty(0)).to(self.device)
    
    def load_weights(self, weights_path, weight_keys='params'):
        load_pretrained_network(self.net, weights_path, weight_keys=weight_keys)
    
    def is_valid_input(self, x):
        if x is not None and self.check_input:
            assert isinstance(x, torch.Tensor), 'Input must be a torch.Tensor'
            assert x.dim() == 4, 'Input must be 4D tensor (B, C, H, W)'
            assert x.shape[1] in [1, 3], 'Input must be RGB or gray image'
            assert x.min() >= 0 and x.max() <= 1, f'Input must be normalized to [0, 1], but got min={x.min():.4f}, max={x.max():.4f}'
    
    def forward(self, target, ref=None, **kwargs):
        device = self.dummy_param.device

        with torch.set_grad_enabled(self.as_loss):

            if self.metric_name == 'fid':
                output = self.net(target, ref, device=device, **kwargs)
            elif self.metric_name == 'inception_score':
                output = self.net(target, device=device, **kwargs)
            else:
                if not torch.is_tensor(target):
                    target = imread2tensor(target, rgb=True)
                    target = target.unsqueeze(0)
                    if self.metric_mode == 'FR':
                        assert ref is not None, 'Please specify reference image for Full Reference metric'
                        ref = imread2tensor(ref, rgb=True)
                        ref = ref.unsqueeze(0)
                
                self.is_valid_input(target)
                self.is_valid_input(ref)

                if self.metric_mode == 'FR':
                    assert ref is not None, 'Please specify reference image for Full Reference metric'
                    output = self.net(target.to(device), ref.to(device), **kwargs)
                elif self.metric_mode == 'NR':
                    output = self.net(target.to(device), **kwargs)

        if self.as_loss:
            if isinstance(output, tuple):
                output = output[0]
            return weight_reduce_loss(output, self.loss_weight, self.loss_reduction)
        else:
            return output
