from PIL import Image
import torch
from torch.utils import data as data

from pyiqa.data.data_util import read_meta_info_file 
from pyiqa.utils.registry import DATASET_REGISTRY
from .base_iqa_feature_dataset import BaseIQAFeatureDataset

import numpy as np
import csv
from os import path as osp

# renyu: 通用的NR预训练特征数据集Data Loader，只是写了下getitem方法，基本方法在baseIQADataset类中定义了
@DATASET_REGISTRY.register()
class GeneralNRFeatureDataset(BaseIQAFeatureDataset):
    """General No Reference dataset with meta info file.
    """

    # renyu: 这里是读取meta_info_file中的文件名，修改成_feature.npy特征文件名后，和mos拼在一起得到"路径-MOS"对
    def init_path_mos(self, opt):
        target_feature_folder = opt['featureroot_target']

        with open(opt['meta_info_file'], 'r') as fin:
            csvreader = csv.reader(fin)
            name_mos = list(csvreader)[1:]

        paths_mos = []
        for item in name_mos:
            img_name, mos = item[:2]
            #img_name = osp.basename(img_name)
            img_name_no_ext = osp.splitext(img_name)[0] 
            feature_name = img_name_no_ext + '_feature.npy'

            feature_path = osp.join(target_feature_folder, feature_name)
            paths_mos.append([feature_path, float(mos)])

        # renyu: LIVE Challenge数据集手动移除前面7个不用的数据
        if opt['name'] == 'livechallenge':
            paths_mos = paths_mos[7:]

        self.paths_mos = paths_mos

    # renyu: 不需要做Transform处理和乘range处理，直接读出来就返回
    def __getitem__(self, index):

        feature_path = self.paths_mos[index][0]
        mos_label = float(self.paths_mos[index][1])

        feature_tensor = np.load(feature_path)
        feature_tensor = np.squeeze(feature_tensor)    # renyu: (1,x,x,x)的格式移除下
        feature_tensor = torch.from_numpy(feature_tensor)
        mos_label_tensor = torch.Tensor([mos_label])
                
        return {'img': feature_tensor, 'mos_label': mos_label_tensor, 'img_path': feature_path}



# renyu: 通用的FR数据集转NR Data Loader，做法很简单，利用read_meta_info_file只读取FR数据标签的NR部分，然后当NR数据集处理MOS即可
#        只是写了下getitem方法，基本方法在baseIQADataset类中定义了
@DATASET_REGISTRY.register()
class GeneralFR2NRFeatureDataset(BaseIQAFeatureDataset):
    """General No Reference to No Reference dataset with meta info file.
    """
    # renyu: 这里是读取meta_info_file中的文件名，注意是忽略FR参考图像部分，修改成_feature.npy特征文件名后，和mos拼在一起得到"路径-MOS"对
    def init_path_mos(self, opt):
        target_feature_folder = opt['featureroot_target']

        with open(opt['meta_info_file'], 'r') as fin:
            csvreader = csv.reader(fin)
            name_mos = list(csvreader)[1:]

        paths_mos = []
        for item in name_mos:
            ref_name, img_name, mos = item[:3]
            #img_name = osp.basename(img_name)
            img_name_no_ext = osp.splitext(img_name)[0] 
            feature_name = img_name_no_ext + '_feature.npy'

            feature_path = osp.join(target_feature_folder, feature_name)
            paths_mos.append([feature_path, float(mos)])

        self.paths_mos = paths_mos

    # renyu: 不需要做Transform处理和乘range处理，直接读出来就返回
    def __getitem__(self, index):

        feature_path = self.paths_mos[index][0]
        mos_label = float(self.paths_mos[index][1])

        feature_tensor = np.load(feature_path)
        feature_tensor = np.squeeze(feature_tensor)    # renyu: (1,x,x,x)的格式移除下
        feature_tensor = torch.from_numpy(feature_tensor)
        mos_label_tensor = torch.Tensor([mos_label])
                
        return {'img': feature_tensor, 'mos_label': mos_label_tensor, 'img_path': feature_path}