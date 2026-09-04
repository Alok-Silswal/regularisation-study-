from .config_node import ConfigNode

config = ConfigNode()

# ---------------------------------------------------------------------------
# Device / cuDNN
# ---------------------------------------------------------------------------

config.device = 'cuda'

config.cudnn = ConfigNode()
config.cudnn.benchmark = True
config.cudnn.deterministic = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

config.dataset = ConfigNode()
config.dataset.name = 'CIFAR10'
config.dataset.dataset_dir = ''
config.dataset.image_size = 32
config.dataset.n_channels = 3
config.dataset.n_classes = 10


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

config.model = ConfigNode()
config.model.type = 'cifar'
config.model.name = 'cnn'
config.model.init_mode = 'kaiming_fan_out'

config.model.cnn = ConfigNode()
config.model.cnn.channels = [32, 64, 128]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

config.train = ConfigNode()

config.train.checkpoint = ''
config.train.resume = False

# Apex is not required for our experiments.
config.train.use_apex = False
config.train.precision = 'O0'

config.train.batch_size = 128
config.train.subdivision = 1

# Optimizer
config.train.optimizer = 'sgd'
config.train.base_lr = 0.1
config.train.momentum = 0.9
config.train.nesterov = True
config.train.weight_decay = 1e-4
config.train.no_weight_decay_on_bn = False
config.train.gradient_clip = 0.0

# Reproducibility
config.train.start_epoch = 0
config.train.seed = 0

# Validation
config.train.val_first = True
config.train.val_period = 1
config.train.val_ratio = 0.1
config.train.use_test_as_val = False

# Output / logging
config.train.output_dir = 'experiments/classification/cnn/baseline_seed0'
config.train.log_period = 100
config.train.checkpoint_period = 10

config.train.use_tensorboard = True


# ---------------------------------------------------------------------------
# TensorBoard
# ---------------------------------------------------------------------------

config.tensorboard = ConfigNode()
config.tensorboard.train_images = False
config.tensorboard.val_images = False
config.tensorboard.model_params = False


# ---------------------------------------------------------------------------
# Learning-rate scheduler
# ---------------------------------------------------------------------------

config.scheduler = ConfigNode()

config.scheduler.epochs = 40

# Warm-up
config.scheduler.warmup = ConfigNode()
config.scheduler.warmup.type = 'none'
config.scheduler.warmup.epochs = 0
config.scheduler.warmup.start_factor = 1e-3
config.scheduler.warmup.exponent = 4

# Main scheduler
config.scheduler.type = 'multistep'
config.scheduler.milestones = [20, 30]
config.scheduler.lr_decay = 0.1
config.scheduler.lr_min_factor = 0.001

# Retained because the scheduler implementation may access these fields.
config.scheduler.T0 = 10
config.scheduler.T_mul = 1.


# ---------------------------------------------------------------------------
# Training data loader
# ---------------------------------------------------------------------------

config.train.dataloader = ConfigNode()
config.train.dataloader.num_workers = 2
config.train.dataloader.drop_last = True
config.train.dataloader.pin_memory = False
config.train.dataloader.non_blocking = False


# ---------------------------------------------------------------------------
# Validation data loader
# ---------------------------------------------------------------------------

config.validation = ConfigNode()
config.validation.batch_size = 256

config.validation.dataloader = ConfigNode()
config.validation.dataloader.num_workers = 2
config.validation.dataloader.drop_last = False
config.validation.dataloader.pin_memory = False
config.validation.dataloader.non_blocking = False


# ---------------------------------------------------------------------------
# Distributed training
# ---------------------------------------------------------------------------

config.train.distributed = False

config.train.dist = ConfigNode()
config.train.dist.backend = 'nccl'
config.train.dist.init_method = 'env://'
config.train.dist.world_size = -1
config.train.dist.node_rank = -1
config.train.dist.local_rank = 0
config.train.dist.use_sync_bn = False


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

config.augmentation = ConfigNode()

# Standard CIFAR-10 preprocessing augmentations
config.augmentation.use_random_crop = True
config.augmentation.use_random_horizontal_flip = True

# Study conditions
config.augmentation.use_cutout = False
config.augmentation.use_random_erasing = False
config.augmentation.use_mixup = False
config.augmentation.use_cutmix = False


# Random Crop
config.augmentation.random_crop = ConfigNode()
config.augmentation.random_crop.padding = 4
config.augmentation.random_crop.fill = 0
config.augmentation.random_crop.padding_mode = 'constant'


# Random Horizontal Flip
config.augmentation.random_horizontal_flip = ConfigNode()
config.augmentation.random_horizontal_flip.prob = 0.5


# Cutout
config.augmentation.cutout = ConfigNode()
config.augmentation.cutout.prob = 1.0
config.augmentation.cutout.mask_size = 16
config.augmentation.cutout.cut_inside = False
config.augmentation.cutout.mask_color = 0


# Random Erasing
config.augmentation.random_erasing = ConfigNode()
config.augmentation.random_erasing.prob = 0.5
config.augmentation.random_erasing.area_ratio_range = [0.02, 0.4]
config.augmentation.random_erasing.min_aspect_ratio = 0.3
config.augmentation.random_erasing.max_attempt = 20


# MixUp
config.augmentation.mixup = ConfigNode()
config.augmentation.mixup.alpha = 1.0


# CutMix
config.augmentation.cutmix = ConfigNode()
config.augmentation.cutmix.alpha = 1.0


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

config.test = ConfigNode()
config.test.checkpoint = ''
config.test.output_dir = ''
config.test.batch_size = 256

config.test.dataloader = ConfigNode()
config.test.dataloader.num_workers = 2
config.test.dataloader.pin_memory = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_default_config():
    return config.clone()