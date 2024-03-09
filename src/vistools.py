import os

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pylab as plt

import tensorflow as tf

from .config import DatasetConfig, UNetDatasetConfig, HookNetDatasetConfig
from .dataset import DatasetBase
from .utils import SlideContainer


class Visualizer:

    """
    HowTo:

        dataset = UNetDataHandlerOneClass(slides_dir,
                                  contours_dir,
                                  'datasets.xls',
                                  UNetDatasetConfig)

        img_id = np.random.choice(dataset.train_img_ids)  # Random image from training set
        slide = dataset.slides[img_id]

        top_left_level_0 = (18000, 18000)
        h_level_0 = 10000
        w_level_0 = 10000

        res_unet = ResUNet(input_shape=(UNetDatasetConfig.PATCH_SIZE, UNetDatasetConfig.PATCH_SIZE, 3),
                           checkpoints_dir=checkpoints_dir,
                           log_dir=logs_dir,
                           name='test')
        model = res_unet.get_model()

        vis = Visualizer()
        vis.plot(model, slide, top_left_level_0, h_level_0, w_level_0)
    """

    def __init__(self,
                 model_type,
                 threshold=0.5,
                 class_name=None,
                 valid_pooling=False,
                 display_level=None,
                 dataset_config=DatasetConfig,
                 overlap_factor=2,
                 display_res=(1080, 1080)):

        assert model_type in ('unet', 'hooknet')
        self.model_type = model_type
        self.model_output_type = None
        self.output_channels = None

        self.threshold = threshold  # For binary models
        self.class_name = class_name  # For binary models
        self.valid_pooling = valid_pooling
        self.display_level = display_level
        self.config = dataset_config

        self.level = None
        self.levels = None
        self.model_input_size = None
        self.display_res = display_res

        self.tumor_idxs = [self.config.LABEL_DICT[name] for name in self.config.TUMOR_NAMES]
        self.tissue_idxs = list(set(list(self.config.LABEL_DICT.values())) - set(self.tumor_idxs))
        self.tissue_idxs.remove(0)
        self.cancer_ind = self.config.CANCER_IND
        self.tissue_ind = self.config.TISSUE_IND

        self.overlap_factor = overlap_factor

    def _update_model_parameters(self, model: tf.keras.Model):
        self.output_channels = model.output_shape[-1]
        if self.output_channels == 3:
            self.model_output_type = 'cancer'
        elif self.output_channels == 1:
            assert self.class_name is not None, "class name is None"
            self.model_output_type = 'binary'
        else:
            self.model_output_type = 'cancer_type'

        if self.model_type == 'hooknet':
            self.model_input_size = model.input_shape[0][1: 3]
        else:
            self.model_input_size = model.input_shape[1: 3]

    def plot_confusion_mat(self,
                           dataset: DatasetBase,
                           model: tf.keras.Model,
                           overlapping=True,
                           normalize='all',
                           v2=False,
                           plot_individual_mats=False):

        # Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.
        # If None, confusion matrix will not be normalized.
        assert normalize in ('all', 'true', 'pred', None)

        if self.model_type == 'hooknet':
            assert hasattr(self.levels, '__iter__'), "self.levels is not an iterable"
        else:
            assert isinstance(self.level, int), "self.level is not an integer"

        self._update_model_parameters(model)

        confusion_matrices = list()

        if self.output_channels == 3:
            labels = ['BG', 'Cancer', 'Tissue']
        else:
            labels = self.config.TUMOR_NAMES.copy()
            labels.insert(0, 'BG')
            labels.append('Tissue')

        for img_id in dataset.test_img_ids:
            print(f'processing slide {img_id}')
            slide = dataset.slides[img_id]

            top_left = (0, 0)
            w, h = slide.level_dimensions[0]

            pred_map, seg_map = self._generate_pred_map(slide,
                                                        model,
                                                        top_left,
                                                        h,
                                                        w,
                                                        overlapping,
                                                        resize=False,
                                                        v2=v2)
            # print(f'pred_map shape: {pred_map.shape} --- seg_map shape: {seg_map.shape}')

            # fig, axes = plt.subplots(ncols=2, figsize=(12, 18))
            # axes[0].imshow(pred_map)
            # axes[1].imshow(seg_map)
            # fig.savefig(f'{img_id}_out.png')

            predictions = pred_map.flatten()
            y_true = seg_map.flatten()
            cm = confusion_matrix(y_true, predictions, labels=list(range(len(labels))), normalize=normalize)

            if plot_individual_mats:
                disp = ConfusionMatrixDisplay(cm, display_labels=labels)

                fig, ax = plt.subplots(figsize=(12, 12))
                disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
                ax.set_title(f'Confusion matrix for test slide {img_id}')
                fig.savefig(f'confusion_mat_test_slide_{img_id}.png')
            confusion_matrices.append(cm)

        if not plot_individual_mats:
            mean_cm = np.mean(confusion_matrices, axis=0)

            disp = ConfusionMatrixDisplay(mean_cm, display_labels=labels)

            fig, ax = plt.subplots(figsize=(12, 12))
            disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
            ax.set_title('Confusion matrix for test slides')
            fig.savefig(f'confusion_mat_test_slides.png')

    def iou_for_slide(self,
                      dataset: DatasetBase,
                      model: tf.keras.Model,
                      img_id):

        self._update_model_parameters(model)

        slide = dataset.slides[img_id]

        if self.model_output_type == 'binary':
            slide_class = slide.image_set_name
            print(f'image set name: {slide_class.lower()}, target class name: {self.class_name.lower()}')
            if slide_class is not None:
                if slide_class.lower() != self.class_name.lower():
                    print(
                        f'slide class: {slide_class.lower()}, model class: {self.class_name.lower()} --- skipping ...')
                    return
            else:
                print(f'Slide {img_id}: image_set_name = None, skipping ...')
                return

        iou = self._get_mean_iou(slide, model, log=True)

        if self.model_output_type == 'binary':
            print(f'IoU for {img_id} ({slide.image_set_name}) ==> {iou[0]}')

        else:

            if self.model_output_type == 'cancer':
                names = ['BG', 'Cancer', 'Tissue']
            else:
                names = UNetDatasetConfig.TUMOR_NAMES.copy()
                names.insert(0, 'BG')
                names.append('Tissue')

            iou_formatted = dict(zip(names, iou))
            print(f'IoU for {img_id} ({slide.image_set_name}) ==> {iou_formatted}')

    def plot_iou_test_slides(self,
                             dataset: DatasetBase,
                             model: tf.keras.Model,
                             only_mean_iou=True,
                             figsize=(14, 22),
                             fontsize=20):

        self._update_model_parameters(model)

        ious = list()
        labels = list()
        for img_id in dataset.test_img_ids:
            print(f'processing slide {img_id}')
            slide = dataset.slides[img_id]

            if self.model_output_type == 'binary':
                slide_class = slide.image_set_name
                if slide_class is not None:
                    if slide_class.lower() != self.class_name.lower():
                        print(f'slide class: {slide_class.lower()}, model class: {self.class_name.lower()} --- skipping')
                        continue
                else:
                    print(f'Slide {img_id}: image_set_name = None, skipping ...')
                    continue

            iou = self._get_mean_iou(slide, model)

            if self.model_output_type == 'binary':
                print(f'IoU for {img_id} ({slide.image_set_name}) ==> {iou[0]}')

                labels.append(img_id)
                ious.append(iou[0])
            else:

                if self.model_output_type == 'cancer':
                    names = ['BG', 'Cancer', 'Tissue']
                else:
                    names = UNetDatasetConfig.TUMOR_NAMES.copy()
                    names.insert(0, 'BG')
                    names.append('Tissue')

                iou_formatted = dict(zip(names, iou))
                print(f'IoU for {img_id} ({slide.image_set_name}) ==> {iou_formatted}')

                labels.append(img_id)
                if only_mean_iou:
                    ious.append(np.mean(iou))
                else:
                    ious.append(list(iou))

        ious = np.array(ious)

        if not only_mean_iou and self.model_output_type != 'binary':
            n_classes = ious.shape[1]
            width = 0.5 / n_classes
            x = np.arange(len(labels))  # the label locations
            fig, ax = plt.subplots(figsize=figsize)
            for c in range(n_classes):
                vals = np.array(ious)[:, c]
                ax.bar(x + width * c, vals, width, label=f'IoU-Class{c}')
            # ax.bar(x - width / 2, ious, width, label='MeanIoU')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('IoU', fontsize=fontsize)
            ax.set_xlabel('Slide ID', fontsize=fontsize)
            ax.set_title('Class-IoU for Each Test Slide', fontsize=fontsize)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=fontsize)
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
            ax.yaxis.set_tick_params(labelsize=fontsize)
            ax.legend()
        else:
            width = 0.5
            x = np.arange(len(labels))  # the label locations
            fig, ax = plt.subplots(figsize=figsize)

            ax.bar(x + width / 2, ious, width, label=f'MeanIoU')
            # ax.bar(x - width / 2, ious, width, label='MeanIoU')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('MeanIoU', fontsize=fontsize)
            ax.set_xlabel('Slide ID', fontsize=fontsize)
            ax.set_title('Mean-IoU for Each Test Slide', fontsize=fontsize)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=fontsize)
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
            ax.yaxis.set_tick_params(labelsize=fontsize)
            ax.legend()

    def plot_whole_slide(self, model, slide, overlapping=True):
        top_left = (0, 0)
        w, h = slide.level_dimensions[0]
        self.plot(model, slide, top_left, h, w, overlapping)

    def plot(self,
             model: tf.keras.Model,
             slide: SlideContainer,
             top_left_level_0,
             h_level_0,
             w_level_0,
             overlapping=True):

        if self.model_type == 'hooknet':
            assert hasattr(self.levels, '__iter__'), "self.levels is not an iterable"
        else:
            assert isinstance(self.level, int), "self.level is not an integer"

        self._update_model_parameters(model)

        pred_map, image, seg_map = self._generate_pred_map(slide,
                                                           model,
                                                           top_left_level_0,
                                                           h_level_0,
                                                           w_level_0,
                                                           overlapping)
        pred_map = np.squeeze(pred_map)
        seg_map = np.squeeze(seg_map).astype(np.int)

        error_map = self.get_error_map(seg_map, pred_map)

        image_name = os.path.basename(slide.image_path)

        if self.model_type == 'unet':
            ls = self.level
        else:
            ls = self.levels
        tmp = f'{image_name}_tl{top_left_level_0[0]}-{top_left_level_0[1]}_h{h_level_0}_h{w_level_0}_' \
              f'{self.model_type}{ls}-output_{self.model_output_type}'

        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(image.numpy() / 255)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.savefig(f'{tmp}_image.png')

        if self.model_output_type == 'binary':
            vmax = 1
        else:
            vmax = self.output_channels - 1

        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(seg_map, vmin=0, vmax=vmax)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_title('GT')
        fig.savefig(f'{tmp}_gt.png')

        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(pred_map, vmin=0, vmax=vmax)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_title('Predicted')
        fig.savefig(f'{tmp}_pred.png')

        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(error_map, vmin=0, vmax=1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_title('Error Map')
        fig.savefig(f'{tmp}_error.png')

    def _generate_pred_map(self,
                           slide: SlideContainer,
                           model: tf.keras.Model,
                           top_left_level0,
                           h_level0,
                           w_level0,
                           overlaping=True,
                           resize=True,
                           v2=False):

        """generates prediction map for given coordinates

        :returns pred_map (w, h, 1), image (w, h, 3), seg_map (w, h, 1)

        Note: if resize is not True, pred_map(w, h) and seg_map(w, h)"""

        assert self.model_output_type is not None, "model_output_type is None"
        assert self.model_input_size is not None, "model_input_size is None"
        assert self.output_channels is not None, "output_channels is None"

        print(f'model output type: {self.model_output_type}')

        if self.model_type == 'unet':
            ds = slide.level_downsamples[self.level]
            xs, ys, w, h = self._generate_level0_grid(ds, top_left_level0, h_level0, w_level0)
        else:
            sorted_levels = sorted(self.levels)
            ds = slide.level_downsamples[sorted_levels[0]]
            min_h = sorted_levels[-1] * self.model_input_size[1]
            min_w = sorted_levels[-1] * self.model_input_size[0]
            xs, ys, w, h = self._generate_level0_grid(ds, top_left_level0, h_level0, w_level0, min_h, min_w)
            # is_binary = False

        # xs, ys, w, h = self._generate_level0_grid(ds, top_left_level0, h_level0, w_level0)

        container_shape = (int(len(ys) * self.model_input_size[1]),
                           int(len(xs) * self.model_input_size[0]))
        if len(xs) * len(ys) < 9:
            print('less than 9 patches available, turning off the overlapping')
            overlaping = False

        if overlaping:
            num_overlap_pixels = int(self.model_input_size[0] * ds / self.overlap_factor)

            new_xs = list()
            for x in xs:
                new_xs.append(x)
                for i in range(1, self.overlap_factor):
                    new_xs.append(x + i * num_overlap_pixels)

            new_ys = list()
            for y in ys:
                new_ys.append(y)
                for i in range(1, self.overlap_factor):
                    new_ys.append(y + i * num_overlap_pixels)

            xs = new_xs[:-(self.overlap_factor - 1)]
            ys = new_ys[:-(self.overlap_factor - 1)]

            print(f'{len(xs) * len(ys)} overlapping patches for model')
        else:
            print(f'{len(xs) * len(ys)} non-overlapping patches for model')

        container = np.zeros(container_shape + (self.output_channels,))
        if self.model_output_type == 'binary':
            container = np.zeros(container_shape + (2,))
        seg_map_container = np.zeros(container_shape)

        if self.model_type == 'unet':
            g = self.gen_unet(slide, xs, ys, model)
        else:
            g = self.gen_hooknet(slide, xs, ys, model)

        for pred, sm, (i, j) in g:
            if not overlaping:
                x = int(i * self.model_input_size[0])
                y = int(j * self.model_input_size[1])
            else:
                x = int(i / self.overlap_factor * self.model_input_size[0])
                y = int(j / self.overlap_factor * self.model_input_size[1])

            # if self.valid_pooling:
            #     pred = cv2.resize(pred, self.model_input_size)
            try:
                if self.model_output_type == 'binary':
                    pred_two_class = np.zeros(pred.shape[:2] + (2,))
                    pred_two_class[:, :, 0] = 1 - pred[:, :, 0]
                    pred_two_class[:, :, 1] = pred[:, :, 0]
                    container[y: y + self.model_input_size[1], x: x + self.model_input_size[0]] += pred_two_class
                else:
                    container[y: y + self.model_input_size[1], x: x + self.model_input_size[0]] += pred
                seg_map_container[y: y + self.model_input_size[1], x: x + self.model_input_size[0]] = sm
                # container_weights[y: y + self.model_input_size[1], x: x + self.model_input_size[0]] += 1
            except Exception as e:
                print(e)
                raise e

        pred_map = container.argmax(axis=-1)

        if self.model_output_type == 'binary':
            class_index = DatasetConfig.LABEL_DICT[self.class_name]
            seg_map = np.zeros_like(seg_map_container)
            seg_map[np.where(seg_map_container == class_index)] = 1
        else:
            if self.model_output_type == 'cancer':
                seg_map = slide.to_binary_maps_cancer(seg_map_container)
                seg_map = seg_map.argmax(axis=-1)
            else:
                if v2:
                    seg_map = slide.to_binary_maps_cancer_type_v2(seg_map_container)
                else:
                    seg_map = slide.to_binary_maps_cancer_type(seg_map_container)
                    seg_map = seg_map.argmax(axis=-1)

        if resize:
            pred_map = tf.image.resize_with_pad(np.expand_dims(pred_map, axis=-1),
                                                self.display_res[0],
                                                self.display_res[1])
            seg_map = tf.image.resize_with_pad(np.expand_dims(seg_map, axis=-1),
                                               self.display_res[0],
                                               self.display_res[1])
            image, _ = self._get_patch(slide, top_left_level0, h, w)
            return pred_map, image, seg_map
        else:
            return np.squeeze(pred_map), np.squeeze(seg_map)

        # image, seg_map = self._get_patch(slide, top_left_level0, h, w)
        # if model_output_type == 'cancer':
        #     seg_map = slide.to_binary_maps_cancer(seg_map)
        # else:
        #     seg_map = slide.to_binary_maps_cancer_type(seg_map)
        # seg_map = seg_map.argmax(axis=-1)

    def _get_mean_iou(self, slide: SlideContainer, model: tf.keras.Model, log=False):

        """returns: np.ndarray of shape (n_classes,)"""

        top_left_level_0 = (0, 0)
        w_level_0, h_level_0 = slide.level_dimensions[0]

        if self.model_type == 'unet':
            ds = slide.level_downsamples[self.level]
            xs, ys, w, h = self._generate_level0_grid(ds, top_left_level_0, h_level_0, w_level_0)
        else:
            sorted_levels = sorted(self.levels)
            ds = slide.level_downsamples[sorted_levels[0]]
            min_h = sorted_levels[-1] * self.model_input_size[1]
            min_w = sorted_levels[-1] * self.model_input_size[0]
            xs, ys, w, h = self._generate_level0_grid(ds, top_left_level_0, h_level_0, w_level_0, min_h, min_w)

        # xs, ys, w, h = self._generate_level0_grid(ds, top_left_level_0, h_level_0, w_level_0)
        print(f'{len(xs) * len(ys)} patches for model')

        if self.model_type == 'unet':
            g = self.gen_unet(slide, xs, ys, model)
        else:
            g = self.gen_hooknet(slide, xs, ys, model)

        ious = list()
        for pred, seg_map, (i, j) in g:
            if self.model_output_type == 'binary':
                seg = np.zeros_like(seg_map, dtype=np.int)
                class_index = DatasetConfig.LABEL_DICT[self.class_name]
                seg[np.where(seg_map == class_index)] = 1
                predictions = (np.squeeze(pred) > self.threshold).astype(np.int)
                iou = self.iou(predictions, seg, 2)

                ious.append(iou)
            else:
                if self.model_output_type == 'cancer':
                    seg = slide.to_binary_maps_cancer(seg_map)
                else:
                    seg = slide.to_binary_maps_cancer_type(seg_map)

                seg = seg.argmax(axis=-1)
                predictions = pred.argmax(axis=-1)
                iou = self.iou(predictions, seg, self.output_channels)

                ious.append(iou)

            if log:
                fig, axes = plt.subplots(ncols=2, figsize=(12, 12))
                axes[0].imshow(seg)
                axes[1].imshow(predictions)

                fig.savefig(f'position({i}, {j})_iou{iou}.png')

        return np.mean(ious, axis=0)

    def _add_labels(self, ax):
        img = np.zeros((140, 10))
        for i in range(14):
            img[int(i * 10): int((i + 1) * 10), :] = i

        label_dict_reversed = {i: j for j, i in self.config.LABEL_DICT.items()}

        ax.imshow(img, vmin=0, vmax=13)
        ax.set_yticks([i * 10 + 5 for i in range(14)])
        ax.set_yticklabels([label_dict_reversed[i] for i in list(label_dict_reversed.keys())])
        ax.xaxis.set_visible(False)

    def _get_patch(self, slide, top_left_level_0, h, w):
        display_ds = h / self.display_res[0]
        best_level = slide.slide.get_best_level_for_downsample(display_ds)
        if self.display_level is not None:
            best_level = self.display_level
        print(f'chosing {best_level} for display level')

        h_best_level = int(h / slide.level_downsamples[best_level])
        w_best_level = int(w / slide.level_downsamples[best_level])
        print(f'h: {h_best_level} -- w: {w_best_level}')

        image = np.array(slide.slide.read_region(location=top_left_level_0,
                                                 level=best_level,
                                                 size=(w_best_level, h_best_level)))[:, :, 0: 3]
        image = tf.image.resize_with_pad(image, self.display_res[0], self.display_res[1])

        seg_map = slide.generate_segmentation_map_for_patch(top_left_level_0,
                                                            (w_best_level, h_best_level),
                                                            best_level)
        seg_map = tf.image.resize_with_pad(np.expand_dims(seg_map, axis=-1),
                                           self.display_res[0], self.display_res[1])
        return image, seg_map

    def _generate_level0_grid(self, ds, top_left_level_0, h_level_0, w_level_0, min_h=None, min_w=None):

        """Generates xs and ys in level0, center-wise, for getting patches from slide in rder to feeding into unet

        :returns xs: list of center x locations, ys: list of center y locations, w: int,, h: int
        """

        if min_h is None:
            min_h = ds * self.model_input_size[1]
            min_w = ds * self.model_input_size[0]
        assert h_level_0 > min_h, f'h_level_0 must be greater than {min_h}'
        assert w_level_0 > min_w, f'w_level_0 must be greater than {min_w}'

        h = int(h_level_0 - (h_level_0 % min_h))
        w = int(w_level_0 - (w_level_0 % min_w))
        print(f'changed h from {h_level_0} to {h}')
        print(f'changed w from {w_level_0} to {w}')

        x_pad = int(self.model_input_size[0] * ds / 2)
        xs = [i + x_pad for i in range(top_left_level_0[0],
                                       top_left_level_0[0] + w - x_pad,
                                       int(self.model_input_size[0] * ds))]  #
        y_pad = int(self.model_input_size[1] * ds / 2)
        ys = [i + y_pad for i in range(top_left_level_0[1],
                                       top_left_level_0[1] + h - y_pad,
                                       int(self.model_input_size[1] * ds))]
        return xs, ys, w, h

    @staticmethod
    def get_error_map(seg_map, pred_map):
        out = np.zeros_like(seg_map)
        wheres = np.where(seg_map == pred_map)
        if np.any(wheres):
            out[wheres] = 1
        return out

    @staticmethod
    def iou(y_pred, y_true, n_classes):
        """y_pred, y_true: np.array of shape(h, w)

        :returns a list of ious for multi classes, and an iou for binary cases, i.e. n_classes == 2
        """

        eps = np.finfo(float).eps

        if n_classes == 2:
            inter = np.sum(y_true * y_pred)
            union = np.sum(y_true + y_pred) - inter
            iou = (inter + eps) / (union + eps)
            return [iou]
        else:
            ious = list()
            for ind in range(n_classes):
                t = (y_true == ind).astype(float)
                p = (y_pred == ind).astype(float)

                inter = np.sum(t * p)
                union = np.sum(t + p) - inter
                iou = (inter + eps) / (union + eps)
                ious.append(iou)
            return ious

    def gen_unet(self, slide: SlideContainer, xs, ys, unet: tf.keras.Model):

        """yields:

         prediction => (self.model_input_size[0], self.model_input_size[1], self.output_channels)
         seg_map => (self.model_input_size[0], self.model_input_size[1])

         """

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                patch, seg_map = slide.get_patch((x, y), self.model_input_size, self.level)
                pred = unet.predict(np.expand_dims(patch, axis=0))[0]

                yield pred, seg_map, (i, j)

    def gen_hooknet(self, slide: SlideContainer, xs, ys, hooknet: tf.keras.Model):

        """yields:

         prediction => (self.model_input_size[0], self.model_input_size[1], self.output_channels)
         seg_map => (self.model_input_size[0], self.model_input_size[1])

         """

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                patches, seg_maps = slide.get_patches_hooknet((x, y), self.model_input_size,
                                                              sorted(self.levels), False, 1.0)
                pred = hooknet.predict([np.expand_dims(patches[0], axis=0),
                                       np.expand_dims(patches[1], axis=0)])[0]
                yield pred, seg_maps[0], (i, j)

    # @staticmethod
    # def gen_unet_binary(slide, xs, ys, level, size, unet):
    #     for i, x in enumerate(xs):
    #         for j, y in enumerate(ys):
    #             patch, _ = slide._get_patch((x, y), size, level)
    #             pred = unet.predict(np.expand_dims(patch, axis=0))[0]
    #             yield pred, (i, j)


class UNetVisualizer(Visualizer):

    def __init__(self,
                 level=3,
                 dataset_config=UNetDatasetConfig,
                 **kwargs):
        super(UNetVisualizer, self).__init__(model_type='unet',
                                             dataset_config=dataset_config,
                                             **kwargs)
        self.level = level


class HookNetVisualizer(Visualizer):

    def __init__(self,
                 levels=(2, 3),
                 dataset_config=HookNetDatasetConfig,
                 **kwargs):
        super(HookNetVisualizer, self).__init__(model_type='hooknet',
                                                dataset_config=dataset_config,
                                                **kwargs)
        self.levels = levels
