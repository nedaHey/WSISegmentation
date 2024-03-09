from typing import List, Dict

import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from ..config import HookNetConfig
from .model import BaseModel
from .utils import mean_iou_multi, dice_coef_loss, crossentropy_dice_loss, jaccard_distance, class_iou,\
    get_weighted_cce, gen_dice


class HookNet(BaseModel):

    def __init__(self,
                 checkpoints_dir,
                 log_dir,
                 name,
                 output_type=HookNetConfig.OUTPUT_TYPE,
                 config=HookNetConfig,
                 class_ious=True):
        super(HookNet, self).__init__(checkpoints_dir, log_dir, name)

        self.config = config

        self.batch_norm = config.BATCH_NORM
        self.dropout = config.DROPOUT
        self.lr = config.LR
        self.input_height, self.input_width = config.INPUT_H, config.INPUT_W
        self.conv_activation = config.CONV_ACTIVATION
        self.kernel_init = config.KERNEL_INIT
        self.kernel_reg = config.KERNEL_REG

        if output_type == 'cancer':
            self.n_classes = 3
        elif output_type == 'cancer_type':
            self.n_classes = self.config.OUTPUT_CHANNELS
        else:
            raise Exception('output_type should be eighter of (cancer, cancer_type)')
        self.output_type = output_type

        self.hooks = (2, 3)
        self.class_ious = class_ious

    def generate_model(self):
        inputs = list()
        outputs = list()

        inp, out, hooks = self._get_unet()
        inputs.append(inp)
        # outputs.append(out)

        inp, out, hooks = self._get_unet(False, hooks)
        inputs.append(inp)
        outputs.append(out)

        # if self.n_classes == 3:
        #     sess_type = 'cancer'
        # else:
        #     sess_type = 'cancer_type'

        model = tfk.Model(inputs, outputs)

        loss = self._get_loss()
        losses = {'target_out': loss}
        metrics = self._get_metrics()
        optimizer = tfk.optimizers.Adam(lr=self.lr)

        model.compile(loss=losses, optimizer=optimizer, metrics=metrics)
        return model

    def _get_loss(self):
        loss_type = self.config.LOSS_TYPE

        if loss_type == 'crossentropy':
            loss = tfk.losses.categorical_crossentropy
        elif loss_type == 'dice_loss':
            loss = dice_coef_loss
        elif loss_type == 'crossentropy_dice_loss':
            loss = crossentropy_dice_loss
        elif loss_type == 'jaccard_distance':
            loss = jaccard_distance
        elif loss_type == 'wcce':
            if self.output_type == 'cancer':
                print('using weighted categorical cross-entropy loss ...')
                weights = list()
                for name in ['BG', 'Cancer', 'Tissue']:
                    loss_weight = self.config.CLASS_WEIGHTS[name]
                    weights.append(loss_weight)
                    print(f'loss weight for {name} = {loss_weight}')
                loss = get_weighted_cce(weights)
            else:
                print('using weighted categorical cross-entropy loss ...')
                weights = list()
                for name in self.config.OUTPUT_NAMES:
                    loss_weight = self.config.CLASS_WEIGHTS[name]
                    weights.append(loss_weight)
                    print(f'loss weight for {name} = {loss_weight}')
                loss = get_weighted_cce(weights)
        elif loss_type == 'gen_dice':
            loss = gen_dice

        else:
            raise Exception('config.LOSS_TYPE not from'
                            ' (crossentropy, dice_loss, crossentropy_dice_loss, jaccard_distance, wcce, gen_dice)')
        return loss

    def _get_metrics(self):
        metrics = [mean_iou_multi(self.n_classes)]
        if self.class_ious:
            if self.output_type == 'cancer':
                metrics.append(class_iou(0, 'BG'))
                metrics.append(class_iou(1, 'Cancer'))
                metrics.append(class_iou(2, 'Tissue'))
            else:
                for i, name in enumerate(self.config.OUTPUT_NAMES):
                    metrics.append(class_iou(i, name))
        return metrics

    def _get_unet(self, is_context=True, hooked=None):
        if not is_context:
            assert hooked is not None, ""

        model_type = 'context'
        if not is_context:
            model_type = 'target'

        input_tensor = tfkl.Input((self.input_height, self.input_width, 3), name=f'{model_type}_input')
        x = tfkl.Lambda(lambda b: b / 255, name=f'{model_type}_normalizer')(input_tensor)
        base_filter_num = 16

        hooks = dict()
        residuals = list()

        for i in range(4):
            dropout_rate = 0.1 * (i // 2 + 1)
            n_filters = 16 * (2 ** i)
            name_prefix = f'{model_type}_encoder_{i + 1}_'

            x = self._conv(x, n_filters, name_prefix + 'conv1')
            if self.dropout:
                x = tfkl.Dropout(dropout_rate, name=name_prefix + 'dropout')(x)
            x = self._conv(x, n_filters, name_prefix + 'conv2')
            residuals.append(x)
            x = tfkl.MaxPooling2D((2, 2), name=name_prefix + 'maxpool')(x)

        x = self._conv(x, base_filter_num * (2 ** 5), f'{model_type}_middle_conv1')
        if self.dropout:
            x = tfkl.Dropout(0.3, name=f'{model_type}_middle_dropout')(x)
        x = self._conv(x, base_filter_num * (2 ** 5), f'{model_type}_middle_conv2')

        # Decoder
        for i, c in enumerate(reversed(residuals)):
            name_prefix = f'{model_type}_decoder_{i + 1}_'

            x = tfkl.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name=name_prefix + 'trans')(x)
            x = tfkl.concatenate([x, c], name=name_prefix + 'concat')
            x = self._conv(x, 128, name_prefix + 'conv1')
            if self.dropout:
                x = tfkl.Dropout(0.2, name=name_prefix + 'dropout')(x)
            x = self._conv(x, 128, name_prefix + 'conv2')

            if i in self.hooks:
                if is_context:
                    hooks[i] = x
                else:
                    crop_size = int(hooked[i].shape[1] - x.shape[1]) / 2
                    context_cropped = tfkl.Cropping2D(int(crop_size))(hooked[i])
                    x = tfkl.concatenate([context_cropped, x], axis=3)

        x = tfkl.Conv2D(self.n_classes, 1, activation=None, name=f'{model_type}_out_conv')(x)
        out = tfkl.Softmax(axis=-1, name=f'{model_type}_out')(x)

        return input_tensor, out, hooks

        # c1 = self._conv(s, 16, 'encoder_1_1')
        # if self.dropout:
        #     c1 = tfkl.Dropout(0.1)(c1)
        # c1 = self._conv(c1, 16, 'encoder_1_2')
        # p1 = tfkl.MaxPooling2D((2, 2))(c1)
        #
        # c2 = self._conv(p1, 32, 'encoder_2_1')
        # if self.dropout:
        #     c2 = tfkl.Dropout(0.1)(c2)
        # c2 = self._conv(c2, 32, 'encoder_2_2')
        # p2 = tfkl.MaxPooling2D((2, 2))(c2)
        #
        # c3 = self._conv(p2, 64, 'encoder_3_1')
        # if self.dropout:
        #     c3 = tfkl.Dropout(0.2)(c3)
        # c3 = self._conv(c3, 64, 'encoder_3_2')
        # p3 = tfkl.MaxPooling2D((2, 2))(c3)
        #
        # c4 = self._conv(p3, 128, 'encoder_4_1')
        # if self.dropout:
        #     c4 = tfkl.Dropout(0.2)(c4)
        # c4 = self._conv(c4, 128, 'encoder_4_2')
        # p4 = tfkl.MaxPooling2D(pool_size=(2, 2))(c4)
        #
        # c5 = self._conv(p4, 256, 'encoder_5_1')
        # if self.dropout:
        #     c5 = tfkl.Dropout(0.3)(c5)
        # c5 = self._conv(c5, 256, 'encoder_5_2')

        # u6 = tfkl.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='decoder_1_trans')(c5)
        # u6 = tfkl.concatenate([u6, c4])
        # c6 = self._conv(u6, 128, 'decoder_1_1')
        # if self.dropout:
        #     c6 = tfkl.Dropout(0.2)(c6)
        # c6 = self._conv(c6, 128, 'decoder_1_2')
        #
        # u7 = tfkl.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='decoder_2_trans')(c6)
        # u7 = tfkl.concatenate([u7, c3])
        # c7 = self._conv(u7, 64, 'decoder_2_1')
        # if self.dropout:
        #     c7 = tfkl.Dropout(0.2)(c7)
        # c7 = self._conv(c7, 64, 'decoder_2_2')
        #
        # u8 = tfkl.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='decoder_3_trans')(c7)
        # u8 = tfkl.concatenate([u8, c2])
        # c8 = self._conv(u8, 32, 'decoder_3_1')
        # if self.dropout:
        #     c8 = tfkl.Dropout(0.1)(c8)
        # c8 = self._conv(c8, 32, 'decoder_3_2')
        #
        # u9 = tfkl.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', name='decoder_4_trans')(c8)
        # u9 = tfkl.concatenate([u9, c1], axis=3)
        # c9 = self._conv(u9, 16, 'decoder_4_1')
        # if self.dropout:
        #     c9 = tfkl.Dropout(0.1)(c9)
        # c9 = self._conv(c9, 16, 'decoder_4_2')
        #
        # x = tfkl.Conv2D(self.n_classes, 1, activation=None, name='out_conv')(c9)
        # out = tfkl.Softmax(axis=-1, name='out')(x)
        #
        # model = tfk.Model(inputs=[inputs], outputs=[out])
        # return model

    def _conv(self, x, filters, name=None):
        x = tfkl.Conv2D(filters, (3, 3),
                        activation=self.conv_activation,
                        kernel_initializer=self.kernel_init,
                        padding='same',
                        kernel_regularizer=self.kernel_reg,
                        name=name)(x)
        if self.batch_norm:
            x = tfkl.BatchNormalization(name=name + '_bn')(x)
        return x


# class HookNetOrg(tfk.Model):
#
#     # TODO: Add normalization to model's input
#
#     def __init__(self,
#                  input_shape=(284, 284, 3),
#                  n_classes=DatasetConfig.N_CLASSES,
#                  hook_indexes=(3, 3),
#                  depth=4,
#                  n_convs=2,
#                  filter_size=3,
#                  n_filters=16,
#                  padding="valid",
#                  batch_norm=True,
#                  activation="relu",
#                  learning_rate=0.000005,
#                  opt_name="adam",
#                  l2_lambda=0.001,
#                  loss_weights=(1.0, 0.2)):
#
#         """
#         Parameters
#         ----------
#         input_shape : List[int]
#             the input shape of the model for both branches
#         n_classes: int
#             the possible number of classes in the output of the model
#         hook_indexes: List[int]
#             the respective depths (starting from 0) of hooking [from, to] in the decoders
#         depth: int
#             the depth of the encoder-decoder branches
#         n_convs: int
#             the number of 2D convolutions per convolutional block
#         filter_size: int
#             the size of the filter in a 2D convolution
#         n_filters: intv
#             the number of starting filters (will be increased and decreased by a factor 2 in each conv block in the encoders and decoders, respectively)
#         padding: str
#             padding type in 2D convolution (either 'same' or 'valid')
#         batch_norm: bool
#             boolean for using batch normalization
#         activation: str
#             activation function after 2D convolution
#         learning_rate: float
#             learning rate of the optimizer
#         opt_name: str
#             optimizer name (either 'sgd' or 'adam')
#         l2_lambda: float
#             l2 value for regulizer
#         loss_weights: bool
#             loss contribution for each branch
#         """
#
#         super().__init__()
#         self._input_shape = input_shape
#         self._n_classes = n_classes
#         self._hook_indexes = {(depth - 1) - hook_indexes[0]: hook_indexes[1]}
#         self._depth = depth
#         self._n_convs = n_convs
#         self._filter_size = filter_size
#         self._n_filters = n_filters
#         self._padding = padding
#         self._batch_norm = batch_norm
#         self._activation = activation
#         self._learning_rate = learning_rate
#         self._opt_name = opt_name
#         self._l2_lambda = l2_lambda
#         self._loss_weights = loss_weights
#
#         # determine multi-loss model from loss weights
#         self._multi_loss = any(loss_weights[1:])
#
#         # set l2 regulizer
#         self._l2 = tfk.regularizers.l2(self._l2_lambda)
#
#         # placeholder for output_shape
#         self._output_shape = []
#
#         # construct model
#         self._construct_hooknet()
#
#     @property
#     def input_shape(self) -> List[int]:
#         """Return the input shape of the model"""
#
#         return self._input_shape
#
#     @property
#     def output_shape(self) -> List[int]:
#         """Return the output shape of the model before flattening"""
#
#         return self._output_shape
#
#     def multi_loss(self) -> bool:
#         return self._multi_loss
#
#     def _construct_hooknet(self) -> None:
#         """Construction of single/multi-loss model with multiple inputs and single/multiple outputs"""
#
#         # declaration of context input
#         input_2 = tfkl.Input(self._input_shape)
#
#         # construction of context branch and context hooks
#         flatten2, context_hooks = self._construct_branch(
#             input_2, reshape_name="reshape_context"
#         )
#
#         # declaration of target inpput
#         input_1 = tfkl.Input(self._input_shape)
#
#         # construction of target branch with context hooks
#         flatten1, _ = self._construct_branch(
#             input_1, context_hooks, reshape_name="reshape_target"
#         )
#
#         # create single/multi loss model
#         if self._multi_loss:
#             self._create_model([input_1, input_2], [flatten1, flatten2])
#         else:
#             self._create_model([input_1, input_2], flatten1)
#
#     def _construct_branch(self, input, in_hooks, reshape_name="reshape_target"):
#
#         """
#         Construction of single branch
#         Parameters
#         ----------
#         input : Input
#             keras Input Tensor
#         in_hooks : Dict
#             A mapping for hooking from the context branch to the target branch
#         reshape_name: str
#             name for Reshape Tensor
#         Returns
#         -------
#         flatten: Tensor
#             last Tensor of the branch
#         out_hooks: Dict
#             mapping for hooking between branches
#         """
#
#         # input
#         net = input
#
#         # encode and retreive residuals
#         net, residuals = self._encode_path(net)
#
#         # mid conv block
#         net = self._conv_block(net, self._n_filters * 2 * (self._depth + 1))
#
#         # decode and retreive hooks
#         net, out_hooks = self._decode_path(net, residuals, in_hooks)
#
#         # softmax output
#         net = tfkl.Conv2D(self._n_classes, 1, activation="softmax")(net)
#
#         # set output shape
#         self._output_shape = tfk.backend.int_shape(net)[1:]
#
#         # Reshape net
#         flatten = tfkl.Reshape(
#             (self.output_shape[0] * self.output_shape[1], self.output_shape[2]),
#             name=reshape_name,
#         )(net)
#
#         # return flatten output and hooks
#         return flatten, out_hooks
#
#     def _encode_path(self, net):
#
#         """
#         Encoder
#         Parameters
#         ----------
#         net: Tensor
#             current Tensor in the model
#         Returns
#         -------
#         net: Tensor
#             current Tensor in the model
#         residuals: List[Tensors]
#             all the Tensors used residuals/skip connections in the decoder part of the model
#         """
#
#         # list for keeping track for residuals/skip connections
#         residuals = []
#
#         # set start filtersize
#         n_filters = self._n_filters
#
#         # loop through depths
#         for b in range(self._depth):
#             # apply convblock
#             net = self._conv_block(net, n_filters)
#
#             # keep Tensor for residual/sip connection
#             residuals.append(net)
#
#             # downsample
#             net = self._downsample(net)
#
#             # increase number of filters with factor 2
#             n_filters *= 2
#
#         return net, residuals
#
#     def _decode_path(self, net, residuals, inhooks={}):
#
#         """
#         Decoder
#         Parameters
#         ----------
#         net: Tensor
#             current Tensor in the model
#         residuals: List[Tensors]
#             all the Tensors used residuals/skip connections in the decoder part of the model
#         in_hooks: Dict
#             mapping for hooking between branches
#         Returns
#         -------
#         net: Tensor
#             current Tensor in the model
#         hooks: Dict
#             mapping between index and Tensor in model for hooking between branches
#         """
#
#         # list for keeping potential hook Tensors
#         outhooks = []
#
#         # set start number of filters of decoder
#         n_filters = self._n_filters * 2 * self._depth
#
#         # loop through depth in reverse
#         for b in reversed(range(self._depth)):
#
#             # hook if hook is available
#             if b in inhooks:
#                 # combine feature maps via merge type
#                 if self._merge_type == "concat":
#                     net = self._concatenator(net, inhooks[b])
#                 else:
#                     net = self._merger(net, inhooks[b])
#
#             # upsample
#             net = self._upsample(net, n_filters)
#
#             # concatenate residuals/skip connections
#             net = self._concatenator(net, residuals[b])
#
#             # apply conv block
#             net = self._conv_block(net, n_filters)
#
#             # set potential hook
#             outhooks.append(net)
#
#             n_filters = n_filters // 2
#
#         # get hooks from potential hooks
#         hooks = {}
#         for shook, ehook in self._hook_indexes.items():
#             hooks[ehook] = outhooks[shook]
#
#         return net, hooks
#
#     def _conv_block(self, net, n_filters, kernel_size=3):
#
#         """
#         Convolutional Block
#         Parameters
#         ----------
#         net: Tensor
#             current Tensor in the model
#         n_filters: int
#             current number of filters
#         kernel_size: int:
#             size of filter in 2d convolution
#         Returns
#         -------
#         net: Tensor
#             current Tensor of the model
#         """
#
#         # loop through number of convolutions in convolution block
#         for n in range(self._n_convs):
#             # apply 2D convolution
#             net = tfkl.Conv2D(
#                 n_filters,
#                 kernel_size,
#                 activation=self._activation,
#                 kernel_initializer="he_normal",
#                 padding=self._padding,
#                 kernel_regularizer=self._l2,
#             )(net)
#
#             # apply batch normalization
#             if self._batch_norm:
#                 net = tfkl.BatchNormalization()(net)
#
#         return net
#
#     def _downsample(self, net):
#
#         """Downsampling via max pooling"""
#
#         return tfkl.MaxPooling2D(pool_size=(2, 2))(net)
#
#     def _upsample(self, net, n_filters):
#
#         """Upsamplign via nearest neightbour interpolation and additional convolution"""
#
#         net = tfkl.UpSampling2D(size=(2, 2))(net)
#         net = tfkl.Conv2D(
#             n_filters,
#             self._filter_size,
#             activation=self._activation,
#             padding=self._padding,
#             kernel_regularizer=self._l2,
#         )(net)
#
#         return net
#
#     def _concatenator(self, net, item):
#
#         """"Concatenate feature maps"""
#
#         # crop feature maps
#         crop_size = int(item.shape[1] - net.shape[1]) / 2
#         item_cropped = tfkl.Cropping2D(int(crop_size))(item)
#
#         return tfkl.concatenate([item_cropped, net], axis=3)
#
#     def _merger(self, net, item):
#
#         """"Combine feature maps"""
#
#         # crop feature maps
#         crop_size = int(item.shape[1] - net.shape[1]) / 2
#         item_cropped = tfkl.Cropping2D(int(crop_size))(item)
#
#         # adapt number of filters via 1x1 convolutional to allow merge
#         current_filters = int(net.shape[-1])
#         item_cropped = tfkl.Conv2D(
#             current_filters, 1, activation=self._activation, padding=self._padding
#         )(item_cropped)
#
#     def _create_model(self, inputs, outputs):
#
#         """
#         Creation of model
#         Parameters
#         ----------
#         inputs: List[Input]
#             inputs to the context and target branch
#         output: List[Reshape]
#             output(s) of the (context) and target branch
#         """
#
#         # initilization of keras model
#         super().__init__(inputs, outputs)
#
#         # set losses and loss weigths
#         losses = (
#             {
#                 "reshape_target": "categorical_crossentropy",
#                 "reshape_context": "categorical_crossentropy",
#             }
#             if self._multi_loss
#             else {"reshape_target": "categorical_crossentropy"}
#         )
#         loss_weights = (
#             {
#                 "reshape_target": self._loss_weights[0],
#                 "reshape_context": self._loss_weights[1],
#             }
#             if self._multi_loss
#             else {"reshape_target": self._loss_weights[0]}
#         )
#
#         # compile model
#         self.compile(
#             optimizer=self._opt(),
#             loss=losses,
#             loss_weights=loss_weights,
#             metrics=["accuracy"],
#         )
#
#     def _opt(self):
#
#         """
#         Set optimizer
#         Returns
#         -------
#         SGD or ADAM optimizer
#         Raises
#         ------
#         ValueError: unsupported optimizer
#         """
#
#         # Set Gradient-descent optimizer
#         if self._opt_name == "sgd":
#             return tfk.optimizers.SGD(lr=self._learning_rate)
#
#         # Set Adam optimizer
#         if self._opt_name == "adam":
#             return tfk.optimizers.Adam(lr=self._learning_rate)
#
#         raise ValueError(f"unsupported optimizer name: {self._opt_name}")
