import tensorflow.keras as tfk


class DatasetConfig:

    BG_THRESHOLD = 0.5
    VAL_DATA_BG_THRESHOLD = 0.8
    PATCH_SIZE = 256
    BATCH_SIZE = 16
    EPOCHS = 50

    LABEL_DICT = {'Bg': 0,
                  'Bone': 1,
                  'Cartilage': 2,
                  'PNST': 3,
                  'Mast Cell Tumor': 4,
                  'Inflamm/Necrosis': 5,
                  'Melanoma': 6,
                  'Plasmacytoma': 7,
                  'SCC': 8,
                  'Trichoblastoma': 9,
                  'Dermis': 10,
                  'Epidermis': 11,
                  'Subcutis': 12,
                  'Histiocytoma': 13}

    TUMOR_NAMES = ['PNST', 'Mast Cell Tumor', 'Melanoma', 'Plasmacytoma', 'SCC', 'Trichoblastoma', 'Histiocytoma']
    N_CLASSES = len(LABEL_DICT)
    N_CANCER = len(TUMOR_NAMES)

    N_TRIALS = 20  # Try N_TRIAL times to extract a patch from a slide which has <BG_THRESHOLD background
    N_ITER_TRAIN_FACTOR = 10  # Statistically get patches from each train slide N_ITER_EPOCH times each epoch

    # UNET_INPUT_LEVEL = max(LEVELS)
    RESIZE_METHOD = 'lanczos5'  # Based on
    # https://stackoverflow.com/questions/384991/what-is-the-best-image-downscaling-algorithm-quality-wise

    CANCER_IND = 1
    TISSUE_IND = -1


class UNetDatasetConfig(DatasetConfig):

    LEVEL = 3
    CLASS_NAME = 'Melanoma'
    MIN_CLASS_AREA = 0.05  # Yields a patch if it has a minimum of 0.05 of CLASS_NAME
    DO_STAIN_AUG = False


class HookNetDatasetConfig(DatasetConfig):

    LEVELS = (2, 3)
    N_ITER_TRAIN_FACTOR = 10


class AugmentationConfig:

    ROTATE_P = 0.2
    ROTATE_LIMIT = 180
    RANDOM_BC_P = 0.2
    RANDOM_GAMMA_P = 0.2
    GAUSS_NOISE_P = 0.2

    # TODO: add stain augmentation


class ModelConfig:

    OUTPUT_CHANNELS = UNetDatasetConfig.N_CANCER + 2  # BG + Cancers + Tissue
    OUTPUT_NAMES = UNetDatasetConfig.TUMOR_NAMES.copy()
    OUTPUT_NAMES.insert(0, 'BG')
    OUTPUT_NAMES.append('Tissue')

    CLASS_WEIGHTS = {
        'PNST': None,
        'Mast Cell Tumor': None,
        'Melanoma': None,
        'Plasmacytoma': None,
        'SCC': None,
        'Trichoblastoma': None,
        'Histiocytoma': None,
        'BG': None,
        'Tissue': None,
        'Cancer': None
    }

    # CLASS_WEIGHTS = {'BG': 0.1, 'Tissue': 0.2, 'Cancer': 0}  # will be used when loss_type==wcce
    # for class_name in DatasetConfig.TUMOR_NAMES:
    #     CLASS_WEIGHTS[class_name] = 0
    #     CLASS_WEIGHTS['Cancer'] += 0


class ResUnetConfig(ModelConfig):

    DECODER_TYPE = 'transpose'  # ['transpose', 'upsample']
    CONV_ACTIVATION = 'relu'
    USE_BATCHNORM = True

    # Note: if OUTPUT_TYPE is 'cancer', LOSS_TYPE will be ignored and crossentropy_dice_loss_multiclass will be
    #   used as the model's loss.
    # Available losses: (binary_crossentropy, dice_loss, crossentropy_dice_loss, jaccard_distance, focal, wcce, gen_dice)
    LOSS_TYPE = 'binary_crossentropy'
    ADAM_LR = 0.001
    IOU_THRESHOLD = 0.5

    OUTPUT_TYPE = 'cancer'  # [binary, cancer, cancer_type]

    USE_KERNEL_REG = False
    L2_LAMBDA = 0.001

    CALLBACK_TYPE = 'reduce_lr'  # [reduce_lr, schedule_lr]


class HookNetConfig(ModelConfig):

    INPUT_H = 256
    INPUT_W = 256
    BATCH_NORM = True
    DROPOUT = False
    CONV_ACTIVATION = 'relu'  # 'elu'
    KERNEL_INIT = 'he_normal'
    KERNEL_REG = None  # tfk.regularizers.l2(0.001)
    LR = 0.001
    LOSS_TYPE = 'crossentropy_dice_loss'  # (crossentropy, crossentropy_dice_loss, dice_loss, jaccard_distance, wcce, gen_dice)

    OUTPUT_TYPE = 'cancer'  # [cancer, cancer_type]


class UNetTrainingConfig:

    JACCARD_DISTANCE_SMOOTH = 100
    OPTIMIZER = tfk.optimizers.Adam(0.001)
    KERNEL_REGULARIZER = tfk.regularizers.l2(0.0001)
    ACTIVATION = 'relu'
    LAST_LAYER_ACTIVATION = 'softmax'
