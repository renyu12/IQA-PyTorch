import logging
import torch
import numpy as np
from os import path as osp
import os

from pyiqa.data import build_dataloader, build_dataset
from pyiqa.models import build_model
from pyiqa.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from pyiqa.utils.options import dict2str, parse_options

from tqdm import tqdm
from pyiqa.archs import build_network
from pyiqa.utils.registry import MODEL_REGISTRY
from pyiqa.models.base_model import BaseModel

# renyu: 特征预提取脚本，参考测试脚本实现，区别在于不需要完整验证获取评分，只需要获取特征并保存为.npy
#        注意这里就是直接读取配置中的所有datasets

# renyu: 这里想复用一些model类的代码来做特征提取，但是作为一个model类和其他模型代码放在一起有点奇怪，直接定义在特征提取脚本中了
@MODEL_REGISTRY.register()
class GeneralIQAFeatureModel(BaseModel):
    """General module to extract features using an IQA network."""

    def __init__(self, opt):
        super(GeneralIQAFeatureModel, self).__init__(opt)

        # define network
        self.net = build_network(opt['network'])
        self.net = self.model_to_device(self.net)
        self.print_network(self.net)
        self.save_feature_path = opt['val'].get('save_feature_path', None)

        # renyu: 保留了统一的预训练模型加载操作，不过SRIQA等模型都是自己在arch里做的预训练模型加载
        load_path = self.opt['path'].get('pretrain_network', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net, load_path, self.opt['path'].get('strict_load', True), param_key)
    
    def feed_data(self, data):
        self.img_input = data['img'].to(self.device)

        if 'mos_label' in data:
            self.gt_mos = data['mos_label'].to(self.device)

        if 'ref_img' in data:
            self.use_ref = True
            self.ref_input = data['ref_img'].to(self.device)
        else:
            self.use_ref = False

        if 'use_ref' in self.opt['train']:
            self.use_ref = self.opt['train']['use_ref']

    def net_forward(self, net):
        if self.use_ref:
            return net(self.img_input, self.ref_input)
        else:
            return net(self.img_input)

    def extract(self):
        self.net.eval()
        with torch.no_grad():
            self.output_feature = self.net_forward(self.net)
        self.net.train()

    def extract_features(self, dataloader):
        dataset_name = dataloader.dataset.opt['name']
        use_pbar = self.opt['val'].get('pbar', False)

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.extract()

            # renyu: 存储预提取的特征，命名规则是仅修改后缀，"原图相对路径/原图名称_feature.npy"
            img_path = val_data['img_path'][0]
            img_dir, img_name_with_ext = osp.split(img_path)
            img_name_no_ext = osp.splitext(img_name_with_ext)[0] 

            # renyu: LIVE数据集有一层相对目录，不太好处理，硬编码一下
            if dataset_name == 'LIVE':
                if osp.sep in img_dir:  
                    parent_dirname, last_segment = osp.split(img_dir)  
                    img_name_no_ext = osp.join(last_segment, img_name_no_ext)
           
            feature_file_path = self.save_feature_path + '/' + img_name_no_ext + '_feature'
            feature_dir, _ = osp.split(feature_file_path)  

            if not osp.exists(feature_dir):  
                os.makedirs(feature_dir)  

            np.save(feature_file_path, self.output_feature.to('cpu').numpy())
            
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Extract {img_name_with_ext:>20}')
        if use_pbar:
            pbar.close()

        log_str = f'Extraction {dataset_name} to {self.save_feature_path} finished.\n'
        log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)


def extract_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    # renyu: 和train一样的解析命令行参数和yaml配置文件，is_train=False的区别是结果存储到results目录而不是experiments目录
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"extract_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='pyiqa', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    pre_extract_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        pre_extract_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        pre_extract_loaders.append(pre_extract_loader)

    # create model
    model = build_model(opt)

    for pre_extract_loader in pre_extract_loaders:
        test_set_name = pre_extract_loader.dataset.opt['name']
        logger.info(f'Pre extracting {test_set_name}...')
        model.extract_features(pre_extract_loader)



if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    extract_pipeline(root_path)
