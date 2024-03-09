import numpy as np
from typing import List, Dict

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow.keras.backend as K

from .config import UNetTrainingConfig, DatasetConfig


class HookNet(tfk.Model):

    # TODO: Add normalization to model's input

    def __init__(self,
                 input_shape=(284, 284, 3),
                 n_classes=DatasetConfig.N_CLASSES,
                 hook_indexes=(3, 3),
                 depth=4,
                 n_convs=2,
                 filter_size=3,
                 n_filters=16,
                 padding="valid",
                 batch_norm=True,
                 activation="relu",
                 learning_rate=0.000005,
                 opt_name="adam",
                 l2_lambda=0.001,
                 loss_weights=(1.0, 0.2)):

        """
        Parameters
        ----------
        input_shape : List[int]
            the input shape of the model for both branches
        n_classes: int
            the possible number of classes in the output of the model
        hook_indexes: List[int]
            the respective depths (starting from 0) of hooking [from, to] in the decoders
        depth: int
            the depth of the encoder-decoder branches
        n_convs: int
            the number of 2D convolutions per convolutional block
        filter_size: int
            the size of the filter in a 2D convolution
        n_filters: intv
            the number of starting filters (will be increased and decreased by a factor 2 in each conv block in the encoders and decoders, respectively)
        padding: str
            padding type in 2D convolution (either 'same' or 'valid')
        batch_norm: bool
            boolean for using batch normalization
        activation: str
            activation function after 2D convolution
        learning_rate: float
            learning rate of the optimizer
        opt_name: str
            optimizer name (either 'sgd' or 'adam')
        l2_lambda: float
            l2 value for regulizer
        loss_weights: bool
            loss contribution for each branch
        """

        super().__init__()
        self._input_shape = input_shape
        self._n_classes = n_classes
        self._hook_indexes = {(depth - 1) - hook_indexes[0]: hook_indexes[1]}
        self._depth = depth
        self._n_convs = n_convs
        self._filter_size = filter_size
        self._n_filters = n_filters
        self._padding = padding
        self._batch_norm = batch_norm
        self._activation = activation
        self._learning_rate = learning_rate
        self._opt_name = opt_name
        self._l2_lambda = l2_lambda
        self._loss_weights = loss_weights

        # determine multi-loss model from loss weights
        self._multi_loss = any(loss_weights[1:])

        # set l2 regulizer
        self._l2 = tfk.regularizers.l2(self._l2_lambda)

        # placeholder for output_shape
        self._output_shape = []

        # construct model
        self._construct_hooknet()

    @property
    def input_shape(self) -> List[int]:
        """Return the input shape of the model"""

        return self._input_shape

    @property
    def output_shape(self) -> List[int]:
        """Return the output shape of the model before flattening"""

        return self._output_shape

    def multi_loss(self) -> bool:
        return self._multi_loss

    def _construct_hooknet(self) -> None:
        """Construction of single/multi-loss model with multiple inputs and single/multiple outputs"""

        # declaration of context input
        input_2 = tfkl.Input(self._input_shape)

        # construction of context branch and context hooks
        flatten2, context_hooks = self._construct_branch(
            input_2, reshape_name="reshape_context"
        )

        # declaration of target inpput
        input_1 = tfkl.Input(self._input_shape)

        # construction of target branch with context hooks
        flatten1, _ = self._construct_branch(
            input_1, context_hooks, reshape_name="reshape_target"
        )

        # create single/multi loss model
        if self._multi_loss:
            self._create_model([input_1, input_2], [flatten1, flatten2])
        else:
            self._create_model([input_1, input_2], flatten1)

    def _construct_branch(self, input, in_hooks, reshape_name="reshape_target"):

        """
        Construction of single branch
        Parameters
        ----------
        input : Input
            keras Input Tensor
        in_hooks : Dict
            A mapping for hooking from the context branch to the target branch
        reshape_name: str
            name for Reshape Tensor
        Returns
        -------
        flatten: Tensor
            last Tensor of the branch
        out_hooks: Dict
            mapping for hooking between branches
        """

        # input
        net = input

        # encode and retreive residuals
        net, residuals = self._encode_path(net)

        # mid conv block
        net = self._conv_block(net, self._n_filters * 2 * (self._depth + 1))

        # decode and retreive hooks
        net, out_hooks = self._decode_path(net, residuals, in_hooks)

        # softmax output
        net = tfkl.Conv2D(self._n_classes, 1, activation="softmax")(net)

        # set output shape
        self._output_shape = tfk.backend.int_shape(net)[1:]

        # Reshape net
        flatten = tfkl.Reshape(
            (self.output_shape[0] * self.output_shape[1], self.output_shape[2]),
            name=reshape_name,
        )(net)

        # return flatten output and hooks
        return flatten, out_hooks

    def _encode_path(self, net):

        """
        Encoder
        Parameters
        ----------
        net: Tensor
            current Tensor in the model
        Returns
        -------
        net: Tensor
            current Tensor in the model
        residuals: List[Tensors]
            all the Tensors used residuals/skip connections in the decoder part of the model
        """

        # list for keeping track for residuals/skip connections
        residuals = []

        # set start filtersize
        n_filters = self._n_filters

        # loop through depths
        for b in range(self._depth):
            # apply convblock
            net = self._conv_block(net, n_filters)

            # keep Tensor for residual/sip connection
            residuals.append(net)

            # downsample
            net = self._downsample(net)

            # increase number of filters with factor 2
            n_filters *= 2

        return net, residuals

    def _decode_path(self, net, residuals, inhooks={}):

        """
        Decoder
        Parameters
        ----------
        net: Tensor
            current Tensor in the model
        residuals: List[Tensors]
            all the Tensors used residuals/skip connections in the decoder part of the model
        in_hooks: Dict
            mapping for hooking between branches
        Returns
        -------
        net: Tensor
            current Tensor in the model
        hooks: Dict
            mapping between index and Tensor in model for hooking between branches
        """

        # list for keeping potential hook Tensors
        outhooks = []

        # set start number of filters of decoder
        n_filters = self._n_filters * 2 * self._depth

        # loop through depth in reverse
        for b in reversed(range(self._depth)):

            # hook if hook is available
            if b in inhooks:
                # combine feature maps via merge type
                if self._merge_type == "concat":
                    net = self._concatenator(net, inhooks[b])
                else:
                    net = self._merger(net, inhooks[b])

            # upsample
            net = self._upsample(net, n_filters)

            # concatenate residuals/skip connections
            net = self._concatenator(net, residuals[b])

            # apply conv block
            net = self._conv_block(net, n_filters)

            # set potential hook
            outhooks.append(net)

            n_filters = n_filters // 2

        # get hooks from potential hooks
        hooks = {}
        for shook, ehook in self._hook_indexes.items():
            hooks[ehook] = outhooks[shook]

        return net, hooks

    def _conv_block(self, net, n_filters, kernel_size=3):

        """
        Convolutional Block
        Parameters
        ----------
        net: Tensor
            current Tensor in the model
        n_filters: int
            current number of filters
        kernel_size: int:
            size of filter in 2d convolution
        Returns
        -------
        net: Tensor
            current Tensor of the model
        """

        # loop through number of convolutions in convolution block
        for n in range(self._n_convs):
            # apply 2D convolution
            net = tfkl.Conv2D(
                n_filters,
                kernel_size,
                activation=self._activation,
                kernel_initializer="he_normal",
                padding=self._padding,
                kernel_regularizer=self._l2,
            )(net)

            # apply batch normalization
            if self._batch_norm:
                net = tfkl.BatchNormalization()(net)

        return net

    def _downsample(self, net):

        """Downsampling via max pooling"""

        return tfkl.MaxPooling2D(pool_size=(2, 2))(net)

    def _upsample(self, net, n_filters):

        """Upsamplign via nearest neightbour interpolation and additional convolution"""

        net = tfkl.UpSampling2D(size=(2, 2))(net)
        net = tfkl.Conv2D(
            n_filters,
            self._filter_size,
            activation=self._activation,
            padding=self._padding,
            kernel_regularizer=self._l2,
        )(net)

        return net

    def _concatenator(self, net, item):

        """"Concatenate feature maps"""

        # crop feature maps
        crop_size = int(item.shape[1] - net.shape[1]) / 2
        item_cropped = tfkl.Cropping2D(int(crop_size))(item)

        return tfkl.concatenate([item_cropped, net], axis=3)

    def _merger(self, net, item):

        """"Combine feature maps"""

        # crop feature maps
        crop_size = int(item.shape[1] - net.shape[1]) / 2
        item_cropped = tfkl.Cropping2D(int(crop_size))(item)

        # adapt number of filters via 1x1 convolutional to allow merge
        current_filters = int(net.shape[-1])
        item_cropped = tfkl.Conv2D(
            current_filters, 1, activation=self._activation, padding=self._padding
        )(item_cropped)

    def _create_model(self, inputs, outputs):

        """
        Creation of model
        Parameters
        ----------
        inputs: List[Input]
            inputs to the context and target branch
        output: List[Reshape]
            output(s) of the (context) and target branch
        """

        # initilization of keras model
        super().__init__(inputs, outputs)

        # set losses and loss weigths
        losses = (
            {
                "reshape_target": "categorical_crossentropy",
                "reshape_context": "categorical_crossentropy",
            }
            if self._multi_loss
            else {"reshape_target": "categorical_crossentropy"}
        )
        loss_weights = (
            {
                "reshape_target": self._loss_weights[0],
                "reshape_context": self._loss_weights[1],
            }
            if self._multi_loss
            else {"reshape_target": self._loss_weights[0]}
        )

        # compile model
        self.compile(
            optimizer=self._opt(),
            loss=losses,
            loss_weights=loss_weights,
            metrics=["accuracy"],
        )

    def _opt(self):

        """
        Set optimizer
        Returns
        -------
        SGD or ADAM optimizer
        Raises
        ------
        ValueError: unsupported optimizer
        """

        # Set Gradient-descent optimizer
        if self._opt_name == "sgd":
            return tfk.optimizers.SGD(lr=self._learning_rate)

        # Set Adam optimizer
        if self._opt_name == "adam":
            return tfk.optimizers.Adam(lr=self._learning_rate)

        raise ValueError(f"unsupported optimizer name: {self._opt_name}")


def get_unet_valid(input_shape=(DatasetConfig.PATCH_SIZE, DatasetConfig.PATCH_SIZE, 3),
                   n_classes=DatasetConfig.N_CLASSES,
                   n_filters=16,
                   depth=4,
                   kernel_size=3,
                   activation=UNetTrainingConfig.ACTIVATION,
                   last_layer_activation=UNetTrainingConfig.LAST_LAYER_ACTIVATION):
    input_tensor = tfkl.Input(shape=input_shape)
    x = tfkl.Lambda(lambda x: x / 255)(input_tensor)

    # list for keeping track for residuals/skip connections
    residuals = []

    # Encoder
    for i in range(depth):
        x = _conv_block(x, n_filters * (2 ** i), kernel_size, activation, 'valid')  # -4 reduction
        residuals.append(x)
        x = tfkl.MaxPooling2D(pool_size=(2, 2))(x)  # /2 reduction

    # Middle convolution block
    x = _conv_block(x, n_filters * (2 ** depth), kernel_size, activation, 'valid')

    # Decoder
    for i in reversed(range(depth)):
        # UpSample
        x = tfkl.UpSampling2D(size=(2, 2))(x)
        x = tfkl.Conv2D(
            n_filters * (2 ** i),
            kernel_size,
            activation=None,
            padding='valid',
            kernel_regularizer=UNetTrainingConfig.KERNEL_REGULARIZER,
        )(x)
        x = tfkl.BatchNormalization()(x)
        x = tfkl.Activation(activation)(x)

        item = residuals[i]
        crop_size = int(item.shape[1] - x.shape[1]) / 2
        item_cropped = tfkl.Cropping2D(int(crop_size))(item)

        x = tfkl.concatenate([item_cropped, x], axis=3)

        x = _conv_block(x, n_filters * (2 ** i), kernel_size, activation, 'valid')

    # softmax output
    if last_layer_activation == 'sigmoid':
        out = tfkl.Conv2D(n_classes, 1, padding='valid', activation='sigmoid')(x)
    else:
        x = tfkl.Conv2D(n_classes, 1, padding='valid', activation=None)(x)
        out = tfkl.Softmax(axis=-1)(x)

    model = tfk.Model(input_tensor, out)
    loss = tfk.losses.categorical_crossentropy
    if last_layer_activation == 'sigmoid':
        loss = tfk.losses.binary_crossentropy

    # metrics = [tfk.metrics.MeanIoU(num_classes=n_classes)] due to the need for flattening
    # the inputs before calculating using this class
    metrics = [mean_iou()]
    model.compile(optimizer=UNetTrainingConfig.OPTIMIZER,
                  loss=loss,
                  metrics=metrics)

    return model


def get_unet(input_shape=(DatasetConfig.PATCH_SIZE, DatasetConfig.PATCH_SIZE, 3),
             n_classes=DatasetConfig.N_CLASSES,
             n_filters=16,
             depth=4,
             kernel_size=3,
             activation=UNetTrainingConfig.ACTIVATION,
             last_layer_activation=UNetTrainingConfig.LAST_LAYER_ACTIVATION):
    input_tensor = tfkl.Input(shape=input_shape)
    x = tfkl.Lambda(lambda x: x / 255)(input_tensor)

    # list for keeping track for residuals/skip connections
    residuals = []

    # Encoder
    for i in range(depth):
        x = _conv_block(x, n_filters * (2 ** i), kernel_size, activation, 'valid')  # -4 reduction
        residuals.append(x)
        x = tfkl.MaxPooling2D(pool_size=(2, 2))(x)  # /2 reduction

    # Middle convolution block
    x = _conv_block(x, n_filters * (2 ** depth), kernel_size, activation, 'same')

    # Decoder
    for i in reversed(range(depth)):
        # UpSample
        x = tfkl.UpSampling2D(size=(2, 2))(x)
        x = tfkl.Conv2D(
            n_filters * (2 ** i),
            kernel_size,
            activation=None,
            padding='same',
            kernel_regularizer=UNetTrainingConfig.KERNEL_REGULARIZER,
        )(x)

        # Concat
        x = tfkl.concatenate([x, residuals[i]], axis=-1)

        # Conv2D Transpose Block
        x = _transconv_block(x, n_filters * (2 ** depth), kernel_size, activation, 'valid')

    # softmax output
    if last_layer_activation == 'sigmoid':
        out = tfkl.Conv2D(n_classes, 1, padding='valid', activation='sigmoid')(x)
    else:
        x = tfkl.Conv2D(n_classes, 1, padding='valid', activation=None)(x)
        out = tfkl.Softmax(axis=-1)(x)

    model = tfk.Model(input_tensor, out)
    loss = tfk.losses.categorical_crossentropy
    if last_layer_activation == 'sigmoid':
        loss = tfk.losses.binary_crossentropy

    # metrics = [tfk.metrics.MeanIoU(num_classes=n_classes)] due to the need for flattening
    # the inputs before calculating using this class
    metrics = [mean_iou()]
    model.compile(optimizer=UNetTrainingConfig.OPTIMIZER,
                  loss=loss,
                  metrics=metrics)

    return model


def _conv_block(x, n_filters, kernel_size, activation, padding):
    x = tfkl.Conv2D(n_filters,
                    kernel_size,
                    activation=None,
                    kernel_initializer="he_normal",
                    padding=padding,
                    kernel_regularizer=UNetTrainingConfig.KERNEL_REGULARIZER)(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Activation(activation)(x)

    x = tfkl.Conv2D(n_filters,
                    kernel_size,
                    activation=None,
                    kernel_initializer="he_normal",
                    padding=padding,
                    kernel_regularizer=UNetTrainingConfig.KERNEL_REGULARIZER)(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Activation(activation)(x)
    return x


def _transconv_block(x, n_filters, kernel_size, activation, padding):
    x = tfkl.Conv2DTranspose(n_filters,
                             kernel_size,
                             activation=None,
                             kernel_initializer="he_normal",
                             padding=padding,
                             kernel_regularizer=UNetTrainingConfig.KERNEL_REGULARIZER)(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Activation(activation)(x)

    x = tfkl.Conv2DTranspose(n_filters,
                             kernel_size,
                             activation=None,
                             kernel_initializer="he_normal",
                             padding=padding,
                             kernel_regularizer=UNetTrainingConfig.KERNEL_REGULARIZER)(x)
    x = tfkl.Activation(activation)(x)
    x = tfkl.BatchNormalization()(x)
    return x


def mean_iou():

    def meaniou(y_true, y_pred):
        return m(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))

    m = tf.metrics.MeanIoU(num_classes=DatasetConfig.N_CLASSES)
    return meaniou


class UNet:

    def __init__(self,
                 n_classes=DatasetConfig.N_CLASSES,
                 input_height=DatasetConfig.PATCH_SIZE,
                 input_width=DatasetConfig.PATCH_SIZE,
                 optimizer=UNetTrainingConfig.OPTIMIZER,
                 use_jaccard_loss=False):
        self.n_classes = n_classes
        self.input_height = input_height
        self.input_width = input_width
        self.use_jaccard_loss = use_jaccard_loss
        self.optimizer = optimizer

    def get_model(self):

        # input image must be RGB(255)

        inputs = tfkl.Input((self.input_height, self.input_width, 3))
        s = tfkl.Lambda(lambda x: x / 255)(inputs)

        c1 = tfkl.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
        c1 = tfkl.Dropout(0.1)(c1)
        c1 = tfkl.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tfkl.MaxPooling2D((2, 2))(c1)

        c2 = tfkl.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tfkl.Dropout(0.1)(c2)
        c2 = tfkl.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = tfkl.MaxPooling2D((2, 2))(c2)

        c3 = tfkl.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tfkl.Dropout(0.2)(c3)
        c3 = tfkl.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = tfkl.MaxPooling2D((2, 2))(c3)

        c4 = tfkl.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = tfkl.Dropout(0.2)(c4)
        c4 = tfkl.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = tfkl.MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = tfkl.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = tfkl.Dropout(0.3)(c5)
        c5 = tfkl.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

        u6 = tfkl.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tfkl.concatenate([u6, c4])
        c6 = tfkl.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = tfkl.Dropout(0.2)(c6)
        c6 = tfkl.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = tfkl.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = tfkl.concatenate([u7, c3])
        c7 = tfkl.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = tfkl.Dropout(0.2)(c7)
        c7 = tfkl.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = tfkl.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = tfkl.concatenate([u8, c2])
        c8 = tfkl.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = tfkl.Dropout(0.1)(c8)
        c8 = tfkl.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = tfkl.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = tfkl.concatenate([u9, c1], axis=3)
        c9 = tfkl.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = tfkl.Dropout(0.1)(c9)
        c9 = tfkl.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

        x = tfkl.Conv2D(self.n_classes, 1, padding='valid', activation=None)(c9)
        out = tfkl.Softmax(axis=-1)(x)

        model = tfk.Model(inputs=[inputs], outputs=[out])
        return model

    def compile(self, model):
        if self.use_jaccard_loss:
            loss = jaccard_distance
        else:
            loss = tfk.losses.categorical_crossentropy
        metrics = [mean_iou()]
        model.compile(optimizer=self.optimizer, loss=loss, metrics=metrics)


# def mean_iou(y_true, y_pred):
#     prec = []
#     for t in np.arange(0.5, 1.0, 0.05):
#         y_pred_ = tf.cast(y_pred > t, tf.int32)
#         m = tf.metrics.MeanIoU(num_classes=DatasetConfig.N_CLASSES)
#         m.update_state(y_true, y_pred)
#         prec.append(m.result().numpy())
#
#         K.get_session().run(tf.local_variables_initializer())
#         with tf.control_dependencies([up_opt]):
#             score = tf.identity(score)
#         prec.append(score)
#     return K.mean(K.stack(prec), axis=0)


def jaccard_distance(y_true, y_pred, smooth=UNetTrainingConfig.JACCARD_DISTANCE_SMOOTH):

    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
    # Returns
        The Jaccard distance between the two tensors.
    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    """

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth