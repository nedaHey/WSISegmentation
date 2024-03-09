import os
from pathlib import Path
from shutil import copyfile

import tensorflow as tf
import tensorflow.keras.callbacks as tfkc


class BaseModel:

    def __init__(self, checkpoints_dir, log_dir, name, callback_type='reduce_lr'):
        assert callback_type in ['reduce_lr', 'schedule_lr']
        self.name = name
        self.checkpoints_dir = Path(checkpoints_dir)
        if not self.checkpoints_dir.exists():
            self.checkpoints_dir.mkdir()
        self.model_checkpoints_dir = self.checkpoints_dir / name
        if not self.model_checkpoints_dir.exists():
            self.model_checkpoints_dir.mkdir()

        self.logs_dir = Path(log_dir)
        if not self.logs_dir.exists():
            self.logs_dir.mkdir()
        self.model_log_dir = self.logs_dir / name
        if not self.model_log_dir.exists():
            self.model_log_dir.mkdir()

        file_dir = Path(__file__).parent.parent
        copyfile(str(file_dir / 'config.py'), str(self.model_log_dir / 'config.py'))

        ch_file_path = str(self.model_checkpoints_dir / 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
        checkpoint_callback = tfkc.ModelCheckpoint(filepath=ch_file_path,
                                                   save_freq='epoch',  # saves model every epoch
                                                   save_weights_only=True,
                                                   verbose=1)
        tensorboard_callback = tfkc.TensorBoard(log_dir=self.model_log_dir,
                                                update_freq='epoch')
        if callback_type == 'schedule_lr':
            lr_callback = tfkc.LearningRateScheduler(self.scheduler)
        else:
            lr_callback = tfkc.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,
                                                 min_delta=0.05, min_lr=0.00005)

        self.callbacks = [checkpoint_callback, tensorboard_callback, lr_callback]
        self.history_ = None
        self.model_ = None
        self.initial_epoch_ = 0

    def generate_model(self):

        """Define the graph and compile the model, and return the compiled model."""

        raise Exception('not implemented!')

    def train(self, train_gen, n_iter_train, val_gen, n_iter_val, epochs):
        model = self.get_model()

        self.history_ = model.fit(train_gen,
                                  steps_per_epoch=n_iter_train,
                                  validation_data=val_gen,
                                  validation_steps=n_iter_val,
                                  epochs=epochs,
                                  callbacks=self.callbacks,
                                  initial_epoch=self.initial_epoch_)

    def get_model(self, is_train=True):

        """Returns last saved model if found, else returns the compiled model.

        Note that the saved models are in SavedModel format and have the configs about training, i.e.
        are compiled models."""

        model = self.generate_model()

        checkpoints = [p for p in self.model_checkpoints_dir.iterdir() if p.match('*.hdf5')]
        # checkpoints = [osjoin(self.model_checkpoints_dir, name) for name in os.listdir(self.model_checkpoints_dir)]
        if checkpoints:
            if is_train:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
                self.initial_epoch_ = int(latest_checkpoint.name.split('.')[1].split('-')[0])
                print("Restoring from ", latest_checkpoint)
                model.load_weights(str(latest_checkpoint))
            else:
                best_checkpoint = min(checkpoints, key=lambda x: float(x.name.split('-')[-1].split('.hdf5')[0]))
                print("Restoring from best model ", best_checkpoint)
                model.load_weights(str(best_checkpoint))
        self.model_ = model
        return model

    @staticmethod
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
