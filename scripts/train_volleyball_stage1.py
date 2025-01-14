import sys
sys.path.append(".")
from train_net import *

cfg=Config('volleyball')
cfg.image_size = 224, 224
cfg.device_list="5,6,7"
cfg.training_stage=1
cfg.stage1_model_path=''
cfg.train_backbone=True

cfg.batch_size=8
cfg.test_batch_size=4
cfg.num_frames=10
cfg.num_before=5
cfg.num_after=4
cfg.train_learning_rate=1e-5
cfg.lr_plan={}
cfg.max_epoch=200
cfg.actions_weights=[[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]
cfg.test_interval_epoch = 5
cfg.test_before_train = False
cfg.out_size = 28, 28
cfg.crop_size = 14, 14

cfg.exp_note='Volleyball_stage1'
train_net(cfg)