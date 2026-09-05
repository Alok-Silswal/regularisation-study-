from yacs.config import CfgNode


def get_default_config():
    config = CfgNode()
    config.device = 'cuda'
    config.seed = 0

    config.dataset = CfgNode()
    config.dataset.name = 'OxfordIIITPet'
    config.dataset.root = ''
    config.dataset.download = False
    config.dataset.image_size = 128
    config.dataset.val_ratio = 0.2
    config.dataset.num_classes = 2

    config.model = CfgNode()
    config.model.base_channels = 16

    config.train = CfgNode()
    config.train.batch_size = 16
    config.train.epochs = 30
    config.train.learning_rate = 1e-3
    config.train.weight_decay = 1e-4
    config.train.num_workers = 2
    config.train.output_dir = 'experiments/segmentation/unet/baseline_seed0'
    config.train.checkpoint_period = 1
    config.train.log_period = 50

    config.augmentation = CfgNode()
    config.augmentation.condition = 'baseline'
    config.augmentation.cutmix_alpha = 1.0
    config.augmentation.cutmix_prob = 1.0
    config.augmentation.cutout_size = 32
    config.augmentation.cutout_prob = 1.0
    config.augmentation.classmix_prob = 1.0
    config.augmentation.horizontal_flip_prob = 0.5

    return config
