import os
from os.path import join as osjoin
from os.path import normpath
from shutil import copyfile

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.callbacks as tfkc

# from .models.utils import mean_iou
# Deprecated, functionality moved to models.model.BaseModel

class Trainer:

    def __init__(self, checkpoints_dir, log_dir, model_name, batch_size):
        self.batch_size = batch_size
        self.model_name = model_name

        self.checkpoints_dir = checkpoints_dir
        if not os.path.exists(self.checkpoints_dir):
            os.mkdir(self.checkpoints_dir)
        self.model_checkpoints_dir = normpath(str(checkpoints_dir / model_name))
        if not os.path.exists(self.model_checkpoints_dir):
            os.mkdir(self.model_checkpoints_dir)

        self.logs_dir = log_dir
        if not os.path.exists(self.logs_dir):
            os.mkdir(self.logs_dir)
        self.model_log_dir = normpath(str(log_dir / model_name))
        if not os.path.exists(self.model_log_dir):
            os.mkdir(self.model_log_dir)

        file_dir = os.path.dirname(os.path.abspath(__file__))
        copyfile(osjoin(file_dir, 'config.py'), osjoin(self.model_log_dir, 'config.py'))

        ch_file_path = normpath(osjoin(self.model_checkpoints_dir, '{epoch:02d}-{val_loss:.2f}'))
        checkpoint_callback = tfkc.ModelCheckpoint(filepath=ch_file_path,
                                                   save_freq='epoch',  # saves model every epoch
                                                   verbose=1)
        tensorboard_callback = tfkc.TensorBoard(log_dir=self.model_log_dir,
                                                update_freq='epoch')
        lr_schedule_callback = tfkc.LearningRateScheduler(self.scheduler)

        self.callbacks = [checkpoint_callback, tensorboard_callback, lr_schedule_callback]
        self.history_ = None

    def train(self, model, train_gen, n_iter_train, val_gen, n_iter_val, epochs):
        model = self._restore_model(model)

        self.history_ = model.fit(train_gen,
                                  steps_per_epoch=n_iter_train,
                                  validation_data=val_gen,
                                  validation_steps=n_iter_val,
                                  epochs=epochs,
                                  callbacks=self.callbacks)

    def _restore_model(self, model):

        """Returns last saved model if found, else returns the compiled model.

        Note that the saved models are in SavedModel format and have the configs about training, i.e.
        are compiled models."""

        checkpoints = [osjoin(self.model_checkpoints_dir, name) for name in os.listdir(self.model_checkpoints_dir)]
        if checkpoints:
            latest_checkpoint = normpath(max(checkpoints, key=os.path.getctime))
            print("Restoring from", latest_checkpoint)
            return tfk.models.load_model(latest_checkpoint, custom_objects={'meaniou': mean_iou()})
        else:
            return model

    @staticmethod
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
