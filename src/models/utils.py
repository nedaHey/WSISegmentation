import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy


# def mean_iou(n_classes):
#
#     def meaniou(y_true, y_pred):
#         m = tf.keras.metrics.MeanIoU(num_classes=n_classes)
#         return m(tf.argmax(y_true, axis=-1),
#                  tf.argmax(y_pred, axis=-1))
#
#     return meaniou


def mean_iou_binary(threshold):

    def mean_iou(y_true, y_pred):
        y_pred = K.cast(K.greater(y_pred, threshold), dtype='float32') # .5 is the threshold
        inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
        union = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) - inter
        return K.mean((inter + K.epsilon()) / (union + K.epsilon()))

    return mean_iou


def mean_iou_multi(n_classes):

    def mean_iou(y_true, y_pred):
        out = 0

        true_max = tf.argmax(y_true, axis=-1)
        pred_max = tf.argmax(y_pred, axis=-1)
        for ind in range(n_classes):
            t = tf.cast(true_max == ind, tf.float32)
            p = tf.cast(pred_max == ind, tf.float32)

            inter = K.sum(K.sum(t * p, axis=2), axis=1)
            union = K.sum(K.sum(t + p, axis=2), axis=1) - inter
            iou = K.mean((inter + K.epsilon()) / (union + K.epsilon()))

            out += iou
        return out / n_classes

    out = mean_iou
    out.name = 'MeanIoU'

    return mean_iou


def class_iou(class_ind, class_name):

    def iou(y_true, y_pred):

        true_max = tf.argmax(y_true, axis=-1)
        pred_max = tf.argmax(y_pred, axis=-1)

        t = tf.cast(true_max == class_ind, tf.float32)
        p = tf.cast(pred_max == class_ind, tf.float32)

        inter = K.sum(K.sum(t * p, axis=2), axis=1)
        union = K.sum(K.sum(t + p, axis=2), axis=1) - inter
        iou = K.mean((inter + K.epsilon()) / (union + K.epsilon()))

        return iou

    out = iou
    out.name = 'iou_' + class_name

    return iou


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1 - (dice_coef(y_true, y_pred))


def get_weighted_cce(weights):

    def wcce(y_true, y_pred):
        w = K.constant(weights)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * w, axis=-1)

    return wcce


def dice_coeff_loss_multiclass(y_true, y_pred):
    out = 0
    for ind in range(y_pred.shape[-1]):
        out += dice_coef_loss(y_true[:, :, :, ind], y_pred[:, :, :, ind])
    return out


def crossentropy_dice_loss_multiclass(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) * 0.5 + dice_coeff_loss_multiclass(y_true, y_pred) * 0.5


def crossentropy_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) * 0.5 + dice_coef_loss(y_true, y_pred) * 0.5


def jaccard_distance(y_true, y_pred):

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

    smooth = 100

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def binary_focal_loss(beta=1., gamma=2.):

    """
    Focal loss is derived from balanced cross entropy, where focal loss adds an extra focus on hard examples in the
    dataset:
        FL(p, p̂) = −[β*(1-p̂)ᵞ*p*log(p̂) + (1-β)*p̂ᵞ*(1−p)*log(1−p̂)]
    When γ = 0, we obtain balanced cross entropy.
    Paper: https://arxiv.org/pdf/1708.02002.pdf
    Used as loss function for binary image segmentation with one-hot encoded masks.
    :param beta: Weight coefficient (float)
    :param gamma: Focusing parameter, γ ≥ 0 (float, default=2.)
    :return: Focal loss (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """

    def loss(y_true, y_pred):

        """
        Computes the focal loss.
        :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
        :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
        :return: Focal loss (tf.Tensor, shape=(<BATCH_SIZE>,))
        """

        f_loss = beta * (1 - y_pred) ** gamma * y_true * K.log(y_pred)  # β*(1-p̂)ᵞ*p*log(p̂)
        f_loss += (1 - beta) * y_pred ** gamma * (1 - y_true) * K.log(1 - y_pred)  # (1-β)*p̂ᵞ*(1−p)*log(1−p̂)
        f_loss = -f_loss  # −[β*(1-p̂)ᵞ*p*log(p̂) + (1-β)*p̂ᵞ*(1−p)*log(1−p̂)]

        # Average over each data point/image in batch
        axis_to_reduce = range(1, K.ndim(f_loss))
        f_loss = K.mean(f_loss, axis=axis_to_reduce)

        return f_loss

    return loss


def gen_dice(y_true, y_pred, eps=1e-6):

    # [b, h, w, classes]
    pred_tensor = tf.nn.softmax(y_pred)
    y_true_shape = tf.shape(y_true)

    # [b, h*w, classes]
    y_true = tf.reshape(y_true, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])
    y_pred = tf.reshape(pred_tensor, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])

    # [b, classes]
    # count how many of each class are present in
    # each image, if there are zero, then assign
    # them a fixed weight of eps
    counts = tf.reduce_sum(y_true, axis=1)
    weights = 1. / (counts ** 2)
    weights = tf.where(tf.math.is_finite(weights), weights, eps)

    multed = tf.reduce_sum(y_true * y_pred, axis=1)
    summed = tf.reduce_sum(y_true + y_pred, axis=1)

    # [b]
    numerators = tf.reduce_sum(weights*multed, axis=-1)
    denom = tf.reduce_sum(weights*summed, axis=-1)
    dices = 1. - 2. * numerators / denom
    dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
    return tf.reduce_mean(dices)


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
