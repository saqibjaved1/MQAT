from .model import PoseModule
from torch import nn
import yaml
from .backbone.darknet53 import *
from .backbone.resnet import *
import os

class Widedepth(nn.Module):
    def __init__(self,name):
        super().__init__()
        with open('/cvlabdata2/home/javed/quantlab/problems/SwissCube/Widedepth/configs/swisscube.yaml', 'r') as f:
            cfg = yaml.load(f,Loader=yaml.Loader)

        if cfg['MODEL']['BACKBONE'] == 'darknet53':
            cfg['MODEL']['FEAT_CHANNELS'] = [0, 0, 256, 512, 1024]
        elif cfg['MODEL']['BACKBONE'] == 'q_resnet18':
            cfg['MODEL']['FEAT_CHANNELS'] = [0, 0, 128, 256, 512]
        else:
            print('Unsupported backbone')
            assert (0)

        cfg['MODEL']['OUT_CHANNEL'] = 256
        cfg['MODEL']['N_CONV'] = 4
        cfg['MODEL']['PRIOR'] = 0.01
        if 'USE_HIGHER_LEVELS' not in cfg['MODEL']:
            cfg['MODEL']['USE_HIGHER_LEVELS'] = True

        cfg['SOLVER']['FOCAL_GAMMA'] = 2.0
        cfg['SOLVER']['FOCAL_ALPHA'] = 0.25
        cfg['SOLVER']['POSITIVE_NUM'] = 10

        cfg['INPUT']['PIXEL_MEAN'] = [0.485, 0.456, 0.406]
        cfg['INPUT']['PIXEL_STD'] = [0.229, 0.224, 0.225]
        cfg['INPUT']['SIZE_DIVISIBLE'] = 32

        if cfg['MODEL']['BACKBONE'] == 'darknet53':
            self.backbone = darknet53(pretrained=True,root = os.path.join("/cvlabdata2/home/javed/quantlab/problems/SwissCube/", ".torch", "models"))
        if cfg['MODEL']['BACKBONE'] == 'q_resnet18':
            self.backbone = resnet18(2, 8, pretrained=True)
        # if cfg['MODEL']['BACKBONE'] == 'resnet18':
        #     self.backbone = resnet18(2, 8, pretrained=True)
        self.model = PoseModule(cfg, self.backbone)
    def forward(self, images, targets=None):
        return self.model.forward(images,targets)
