import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow.keras.models as tfkm

from classification_models.keras import Classifiers  # pip install image-classifiers

from ..config import ResUnetConfig
from .utils import mean_iou_multi, jaccard_distance, mean_iou_binary, dice_coef_loss, crossentropy_dice_loss,\
    binary_focal_loss, class_iou, get_weighted_cce, gen_dice
from .model import BaseModel


class ResUNet(BaseModel):

    def __init__(self,
                 input_shape,
                 checkpoints_dir,
                 log_dir,
                 name,
                 output_type=ResUnetConfig.OUTPUT_TYPE,
                 config=ResUnetConfig,
                 class_ious=True):
        super().__init__(checkpoints_dir, log_dir, name, config.CALLBACK_TYPE)

        self.class_ious = class_ious
        self.config = config
        self.output_type = output_type

        decoder_type = config.DECODER_TYPE
        assert decoder_type in ('transpose', 'upsample')

        self.skip_layer_names = ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0')
        self.decoder_filters = (256, 128, 64, 32, 16)
        self.n_upsample_blocks = len(self.decoder_filters)
        if decoder_type == 'transpose':
            self.decoder_block = self.decoder_transpose_block
        else:
            self.decoder_block = self.decoder_upsampling_block

        self.input_shape = input_shape

        # Regularization Params
        if self.config.USE_KERNEL_REG:
            self.kernel_regularizer = tfk.regularizers.l2(self.config.L2_LAMBDA)
        else:
            self.kernel_regularizer = None

    def generate_model(self):
        use_batchnorm = self.config.USE_BATCHNORM

        resnet_18, _ = Classifiers.get('resnet18')  # For resnet: preprocessed = input
        backbone = resnet_18(input_shape=self.input_shape, weights='imagenet', include_top=False)
        input_ = backbone.input
        x = backbone.output

        # extract skip connections
        skips = ([backbone.get_layer(name=name).output for name in self.skip_layer_names])

        # building decoder blocks
        for i in range(self.n_upsample_blocks):

            if i < len(skips):
                skip = skips[i]
            else:
                skip = None

            x = self.decoder_block(self.decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)

        # model head (define number of output classes)

        if self.output_type == 'binary':
            last_layer_activation = 'sigmoid'
            last_layer_kernels = 1
        elif self.output_type == 'cancer':
            last_layer_activation = 'softmax'
            last_layer_kernels = 3
        elif self.output_type == 'cancer_type':
            last_layer_activation = 'softmax'
            last_layer_kernels = self.config.OUTPUT_CHANNELS
        else:
            raise Exception('output_type should be of (binary, cancer, cancer_type')

        x = tfkl.Conv2D(
            filters=last_layer_kernels,
            kernel_size=(3, 3),
            padding='same',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            name='final_conv',
            kernel_regularizer=self.kernel_regularizer
        )(x)
        x = tfkl.Activation(last_layer_activation, name='output_tensor')(x)

        # create keras model instance
        model = tfkm.Model(input_, x)

        loss = self._get_loss(self.output_type)
        metrics = self._get_metrics(self.output_type)

        optimizer = tfk.optimizers.Adam(lr=self.config.ADAM_LR)
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)

        return model

    def _get_loss(self, output_type):
        if self.config.LOSS_TYPE == 'binary_crossentropy':
            assert output_type == 'binary'
            loss = tfk.losses.binary_crossentropy

        elif self.config.LOSS_TYPE == 'dice_loss':
            loss = dice_coef_loss

        elif self.config.LOSS_TYPE == 'crossentropy_dice_loss':
            loss = crossentropy_dice_loss

        elif self.config.LOSS_TYPE == 'jaccard_distance':
            loss = jaccard_distance

        elif self.config.LOSS_TYPE == 'focal':
            loss = binary_focal_loss()

        elif self.config.LOSS_TYPE == 'wcce':
            print('using weighted categorical cross-entropy loss ...')
            weights = list()

            if output_type == 'cancer':
                class_names = ['BG', 'Cancer', 'Tissue']

            elif output_type == 'cancer_type':
                class_names = self.config.OUTPUT_NAMES

            else:
                raise Exception('output type should be either of (cancer, cancer_type) for wcce loss')

            for name in class_names:
                loss_weight = self.config.CLASS_WEIGHTS[name]
                weights.append(loss_weight)
                print(f'loss weight for {name} = {loss_weight}')
            loss = get_weighted_cce(weights)

        elif self.config.LOSS_TYPE == 'gen_dice':
            loss = gen_dice

        else:
            raise Exception('config.LOSS_TYPE not from'
                            ' [binary_crossentropy, dice_loss, crossentropy_dice_loss, jaccard_distance, focal, wcce, gen_dice]')

        return loss

    def _get_metrics(self, output_type):
        if output_type == 'binary':
            metrics = [mean_iou_binary(self.config.IOU_THRESHOLD)]

        elif output_type == 'cancer':
            metrics = [mean_iou_multi(3)]
            if self.class_ious:
                metrics.append(class_iou(0, 'BG'))
                metrics.append(class_iou(1, 'Cancer'))
                metrics.append(class_iou(2, 'Tissue'))

        elif output_type == 'cancer_type':
            n_classes = self.config.OUTPUT_CHANNELS
            metrics = [mean_iou_multi(n_classes)]
            if self.class_ious:
                for i, name in enumerate(self.config.OUTPUT_NAMES):
                    metrics.append(class_iou(i, name))

        else:
            raise Exception('output_type should be one of (binary, cancer, cancer_type)')

        return metrics

    def conv_bn_relu(self, filters, use_batchnorm, kernel_size=3, name=None):

        def wrapper(input_tensor):
            return self.conv2d_bn(
                filters,
                kernel_size=kernel_size,
                activation=self.config.CONV_ACTIVATION,
                kernel_initializer='he_uniform',
                padding='same',
                use_batchnorm=use_batchnorm,
                name=name,
                kernel_regularizer=self.kernel_regularizer
            )(input_tensor)

        return wrapper

    def decoder_upsampling_block(self, filters, stage, use_batchnorm=False):
        up_name = 'decoder_stage{}_upsampling'.format(stage)
        conv1_name = 'decoder_stage{}a'.format(stage)
        conv2_name = 'decoder_stage{}b'.format(stage)
        concat_name = 'decoder_stage{}_concat'.format(stage)

        def wrapper(input_tensor, skip=None):
            x = tfkl.UpSampling2D(size=2, name=up_name)(input_tensor)

            if skip is not None:
                x = tfkl.Concatenate(axis=3, name=concat_name)([x, skip])

            x = self.conv_bn_relu(filters, use_batchnorm, name=conv1_name)(x)
            x = self.conv_bn_relu(filters, use_batchnorm, name=conv2_name)(x)

            return x

        return wrapper

    def decoder_transpose_block(self, filters, stage, use_batchnorm=False):
        transp_name = 'decoder_stage{}a_transpose'.format(stage)
        bn_name = 'decoder_stage{}a_bn'.format(stage)
        relu_name = 'decoder_stage{}a_relu'.format(stage)
        conv_block_name = 'decoder_stage{}b'.format(stage)
        concat_name = 'decoder_stage{}_concat'.format(stage)

        def layer(input_tensor, skip=None):

            x = tfkl.Conv2DTranspose(
                filters,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same',
                name=transp_name,
                use_bias=not use_batchnorm,
                kernel_regularizer=self.kernel_regularizer
            )(input_tensor)

            if use_batchnorm:
                x = tfkl.BatchNormalization(name=bn_name)(x)

            x = tfkl.Activation(self.config.CONV_ACTIVATION, name=relu_name)(x)

            if skip is not None:
                x = tfkl.Concatenate(axis=3, name=concat_name)([x, skip])

            x = self.conv_bn_relu(filters, use_batchnorm, name=conv_block_name)(x)

            return x

        return layer

    @staticmethod
    def conv2d_bn(
            filters,
            kernel_size,
            strides=(1, 1),
            padding='valid',
            data_format=None,
            dilation_rate=(1, 1),
            activation=None,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            use_batchnorm=False,
            name=None
    ):
        """Extension of Conv2D layer with batchnorm"""

        conv_name, act_name, bn_name = None, None, None
        block_name = name

        if block_name is not None:
            conv_name = block_name + '_conv'

        if block_name is not None and activation is not None:
            act_str = activation.__name__ if callable(activation) else str(activation)
            act_name = block_name + '_' + act_str

        if block_name is not None and use_batchnorm:
            bn_name = block_name + '_bn'

        def wrapper(input_tensor):

            x = tfkl.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                activation=None,
                use_bias=not (use_batchnorm),
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                name=conv_name,
            )(input_tensor)

            if use_batchnorm:
                x = tfkl.BatchNormalization(name=bn_name)(x)

            if activation:
                x = tfkl.Activation(activation, name=act_name)(x)

            return x

        return wrapper

# def get_unet_valid(input_shape=(DatasetConfig.PATCH_SIZE, DatasetConfig.PATCH_SIZE, 3),
#                    n_classes=DatasetConfig.N_CLASSES,
#                    n_filters=16,
#                    depth=4,
#                    kernel_size=3,
#                    activation=UNetTrainingConfig.ACTIVATION,
#                    last_layer_activation=UNetTrainingConfig.LAST_LAYER_ACTIVATION):
#     input_tensor = tfkl.Input(shape=input_shape)
#     x = tfkl.Lambda(lambda x: x / 255)(input_tensor)
#
#     # list for keeping track for residuals/skip connections
#     residuals = []
#
#     # Encoder
#     for i in range(depth):
#         x = _conv_block(x, n_filters * (2 ** i), kernel_size, activation, 'valid')  # -4 reduction
#         residuals.append(x)
#         x = tfkl.MaxPooling2D(pool_size=(2, 2))(x)  # /2 reduction
#
#     # Middle convolution block
#     x = _conv_block(x, n_filters * (2 ** depth), kernel_size, activation, 'valid')
#
#     # Decoder
#     for i in reversed(range(depth)):
#         # UpSample
#         x = tfkl.UpSampling2D(size=(2, 2))(x)
#         x = tfkl.Conv2D(
#             n_filters * (2 ** i),
#             kernel_size,
#             activation=None,
#             padding='valid',
#             kernel_regularizer=UNetTrainingConfig.KERNEL_REGULARIZER,
#         )(x)
#         x = tfkl.BatchNormalization()(x)
#         x = tfkl.Activation(activation)(x)
#
#         item = residuals[i]
#         crop_size = int(item.shape[1] - x.shape[1]) / 2
#         item_cropped = tfkl.Cropping2D(int(crop_size))(item)
#
#         x = tfkl.concatenate([item_cropped, x], axis=3)
#
#         x = _conv_block(x, n_filters * (2 ** i), kernel_size, activation, 'valid')
#
#     # softmax output
#     if last_layer_activation == 'sigmoid':
#         out = tfkl.Conv2D(n_classes, 1, padding='valid', activation='sigmoid')(x)
#     else:
#         x = tfkl.Conv2D(n_classes, 1, padding='valid', activation=None)(x)
#         out = tfkl.Softmax(axis=-1)(x)
#
#     model = tfk.Model(input_tensor, out)
#     loss = tfk.losses.categorical_crossentropy
#     if last_layer_activation == 'sigmoid':
#         loss = tfk.losses.binary_crossentropy
#
#     # metrics = [tfk.metrics.MeanIoU(num_classes=n_classes)] due to the need for flattening
#     # the inputs before calculating using this class
#     metrics = [mean_iou()]
#     model.compile(optimizer=UNetTrainingConfig.OPTIMIZER,
#                   loss=loss,
#                   metrics=metrics)
#
#     return model
#
#
# def get_unet(input_shape=(DatasetConfig.PATCH_SIZE, DatasetConfig.PATCH_SIZE, 3),
#              n_classes=DatasetConfig.N_CLASSES,
#              n_filters=16,
#              depth=4,
#              kernel_size=3,
#              activation=UNetTrainingConfig.ACTIVATION,
#              last_layer_activation=UNetTrainingConfig.LAST_LAYER_ACTIVATION):
#     input_tensor = tfkl.Input(shape=input_shape)
#     x = tfkl.Lambda(lambda x: x / 255)(input_tensor)
#
#     # list for keeping track for residuals/skip connections
#     residuals = []
#
#     # Encoder
#     for i in range(depth):
#         x = _conv_block(x, n_filters * (2 ** i), kernel_size, activation, 'valid')  # -4 reduction
#         residuals.append(x)
#         x = tfkl.MaxPooling2D(pool_size=(2, 2))(x)  # /2 reduction
#
#     # Middle convolution block
#     x = _conv_block(x, n_filters * (2 ** depth), kernel_size, activation, 'same')
#
#     # Decoder
#     for i in reversed(range(depth)):
#         # UpSample
#         x = tfkl.UpSampling2D(size=(2, 2))(x)
#         x = tfkl.Conv2D(
#             n_filters * (2 ** i),
#             kernel_size,
#             activation=None,
#             padding='same',
#             kernel_regularizer=UNetTrainingConfig.KERNEL_REGULARIZER,
#         )(x)
#
#         # Concat
#         x = tfkl.concatenate([x, residuals[i]], axis=-1)
#
#         # Conv2D Transpose Block
#         x = _transconv_block(x, n_filters * (2 ** depth), kernel_size, activation, 'valid')
#
#     # softmax output
#     if last_layer_activation == 'sigmoid':
#         out = tfkl.Conv2D(n_classes, 1, padding='valid', activation='sigmoid')(x)
#     else:
#         x = tfkl.Conv2D(n_classes, 1, padding='valid', activation=None)(x)
#         out = tfkl.Softmax(axis=-1)(x)
#
#     model = tfk.Model(input_tensor, out)
#     loss = tfk.losses.categorical_crossentropy
#     if last_layer_activation == 'sigmoid':
#         loss = tfk.losses.binary_crossentropy
#
#     # metrics = [tfk.metrics.MeanIoU(num_classes=n_classes)] due to the need for flattening
#     # the inputs before calculating using this class
#     metrics = [mean_iou()]
#     model.compile(optimizer=UNetTrainingConfig.OPTIMIZER,
#                   loss=loss,
#                   metrics=metrics)
#
#     return model
#
#
# class UNet:
#
#     def __init__(self,
#                  n_classes=DatasetConfig.N_CLASSES,
#                  input_height=DatasetConfig.PATCH_SIZE,
#                  input_width=DatasetConfig.PATCH_SIZE,
#                  optimizer=UNetTrainingConfig.OPTIMIZER,
#                  use_jaccard_loss=False):
#         self.n_classes = n_classes
#         self.input_height = input_height
#         self.input_width = input_width
#         self.use_jaccard_loss = use_jaccard_loss
#         self.optimizer = optimizer
#
#     def get_model(self):
#
#         # input image must be RGB(255)
#
#         inputs = tfkl.Input((self.input_height, self.input_width, 3))
#         s = tfkl.Lambda(lambda x: x / 255)(inputs)
#
#         c1 = tfkl.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
#         c1 = tfkl.Dropout(0.1)(c1)
#         c1 = tfkl.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
#         p1 = tfkl.MaxPooling2D((2, 2))(c1)
#
#         c2 = tfkl.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
#         c2 = tfkl.Dropout(0.1)(c2)
#         c2 = tfkl.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
#         p2 = tfkl.MaxPooling2D((2, 2))(c2)
#
#         c3 = tfkl.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
#         c3 = tfkl.Dropout(0.2)(c3)
#         c3 = tfkl.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
#         p3 = tfkl.MaxPooling2D((2, 2))(c3)
#
#         c4 = tfkl.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
#         c4 = tfkl.Dropout(0.2)(c4)
#         c4 = tfkl.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
#         p4 = tfkl.MaxPooling2D(pool_size=(2, 2))(c4)
#
#         c5 = tfkl.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
#         c5 = tfkl.Dropout(0.3)(c5)
#         c5 = tfkl.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)
#
#         u6 = tfkl.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
#         u6 = tfkl.concatenate([u6, c4])
#         c6 = tfkl.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
#         c6 = tfkl.Dropout(0.2)(c6)
#         c6 = tfkl.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)
#
#         u7 = tfkl.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
#         u7 = tfkl.concatenate([u7, c3])
#         c7 = tfkl.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
#         c7 = tfkl.Dropout(0.2)(c7)
#         c7 = tfkl.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)
#
#         u8 = tfkl.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
#         u8 = tfkl.concatenate([u8, c2])
#         c8 = tfkl.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
#         c8 = tfkl.Dropout(0.1)(c8)
#         c8 = tfkl.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)
#
#         u9 = tfkl.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
#         u9 = tfkl.concatenate([u9, c1], axis=3)
#         c9 = tfkl.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
#         c9 = tfkl.Dropout(0.1)(c9)
#         c9 = tfkl.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)
#
#         x = tfkl.Conv2D(self.n_classes, 1, padding='valid', activation=None)(c9)
#         out = tfkl.Softmax(axis=-1)(x)
#
#         model = tfk.Model(inputs=[inputs], outputs=[out])
#         return model
#
#     def compile(self, model):
#         if self.use_jaccard_loss:
#             loss = jaccard_distance
#         else:
#             loss = tfk.losses.categorical_crossentropy
#         metrics = [mean_iou()]
#         model.compile(optimizer=self.optimizer, loss=loss, metrics=metrics)
