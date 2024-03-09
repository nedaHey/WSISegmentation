"""How to use:

target_dir = Path('tissue_slides')
if not target_dir.is_dir():
    os.mkdir(target_dir)

handler = DatasetHandler(target_dir)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

from exact_sync.v1.api.annotations_api import AnnotationsApi
from exact_sync.v1.api.images_api import ImagesApi
from exact_sync.v1.api.image_sets_api import ImageSetsApi
from exact_sync.v1.api.annotation_types_api import AnnotationTypesApi
from exact_sync.v1.api.products_api import ProductsApi

from exact_sync.v1.configuration import Configuration
from exact_sync.v1.api_client import ApiClient

from .utils import SlideContainer
from .config import DatasetConfig, UNetDatasetConfig, HookNetDatasetConfig


class DatasetBase:

    def __init__(self,
                 slides_dir,
                 contours_dir,
                 subsets_file_path,
                 patch_size,
                 n_images=None):
        """:arg config: UNetDatasetConfig"""

        self.tumor_names = ['PNST', 'Mast Cell Tumor', 'Melanoma', 'Plasmacytoma', 'SCC', 'Trichoblastoma',
                            'Histiocytoma']

        self.target_dir = slides_dir
        self.contours_dir = contours_dir
        self.patch_size = patch_size

        self.image_sets_api = None
        self.annotations_api = None
        self.annotation_types_api = None
        self.images_api = None
        self.product_api = None
        self.image_sets = None
        self.images_list = None
        self.annotation_types = {}
        self._configure(n_images)

        self.df = self._load_subsets(subsets_file_path)

        print('generating slides ...')

        self.train_img_ids = list()
        self.val_img_ids = list()
        self.test_img_ids = list()
        self.slides = dict()
        self._generate_slides()
        print('{} slides generated.'.format(len(self.slides)))

        self.n_iter_val = None

    def _configure(self, n_images):

        """Pass n_images=None to consider all images."""

        configuration = Configuration()
        configuration.username = 'DemoCanineSkinTumors'
        configuration.password = 'demodemo'
        configuration.host = "https://exact.cs.fau.de/"

        client = ApiClient(configuration)

        self.image_sets_api = ImageSetsApi(client)
        self.annotations_api = AnnotationsApi(client)
        self.annotation_types_api = AnnotationTypesApi(client)
        self.images_api = ImagesApi(client)
        self.product_api = ProductsApi(client)

        image_sets = self.image_sets_api.list_image_sets().results
        self.image_sets = image_sets
        self.images_list = list()

        for image_set in image_sets:
            images_list = self.images_api.list_images(image_set=image_set.id, pagination=False).results

            if n_images is None:
                self.images_list.extend(images_list)
            else:
                self.images_list.extend(images_list[:n_images])

            for product in image_set.product_set:
                for anno_type in self.annotation_types_api.list_annotation_types(product=product).results:
                    self.annotation_types[anno_type.id] = anno_type

    def _generate_slides(self):
        for image in self.images_list:
            if image.filename not in self.df['name'].values:
                print(f'image {image.filename} not found in subsets file, skipping.')
                continue
            image_path = self.target_dir / image.name
            image_set_name = [i.name for i in self.image_sets if i.id == image.image_set]
            try:
                slide = SlideContainer(image.id,
                                       image_path,
                                       self.images_api,
                                       self.annotations_api,
                                       DatasetConfig.LABEL_DICT,
                                       self.annotation_types,
                                       self.contours_dir)
                if image_set_name:

                    if image_set_name[0] == 'Peripheral nerve sheath tumor':
                        slide.image_set_name = 'PNST'

                    elif image_set_name[0] == 'Squamous cell carcinoma':
                        slide.image_set_name = 'SCC'

                    elif image_set_name[0] == 'Mast cell tumor':
                        slide.image_set_name = 'Mast Cell Tumor'

                    else:
                        slide.image_set_name = image_set_name[0]

                    # slide.image_set_name = image_set_name[0]
            except Exception as e:
                print('skipping {} due to {}'.format(image.name, e))
            else:
                if slide.slide.level_count < 4:
                    print('skipping {} due to absence of level 3.'.format(image.name))
                    continue
                subset = self.df[self.df['name'] == image.filename]['subset'].values[0]
                if subset == 'val':
                    self.val_img_ids.append(image.id)
                elif subset == 'test':
                    self.test_img_ids.append(image.id)
                else:
                    self.train_img_ids.append(image.id)
                self.slides[image.id] = slide
        print(f'{len(self.train_img_ids)} train slides, {len(self.val_img_ids)} validation slides.')

    @staticmethod
    def _load_subsets(file_path):

        def data_subset(row):
            return row[0].split(';')[-1]

        def img_name(row):
            return row[0].split(';')[0]

        df = pd.read_csv(file_path)
        df['subset'] = df.apply(data_subset, axis=1)
        df['name'] = df.apply(img_name, axis=1)
        df.drop('Slide;Dataset', inplace=True, axis=1)
        return df

    @staticmethod
    def _get_random_location(slide, size, sorted_levels):

        """Returns a random location based on coordinates of level0 and
        takes into account the limits of the last level"""

        downsample = slide.level_downsamples[sorted_levels[-1]]
        dims_level0 = slide.dimensions
        pad_x = np.ceil(size[0] / 2 * downsample).astype(int) + int(downsample)
        pad_y = np.ceil(size[1] / 2 * downsample).astype(int) + int(downsample)
        x_allowed = np.arange(pad_x, dims_level0[0] - pad_x)
        y_allowed = np.arange(pad_y, dims_level0[1] - pad_y)

        x = np.random.choice(x_allowed)
        y = np.random.choice(y_allowed)

        location = (x, y)

        return location


class DatasetHandlerHooknetCancer(DatasetBase):

    def __init__(self,
                 slides_dir,
                 contours_dir,
                 subsets_file_path,
                 config=HookNetDatasetConfig,
                 n_images=None):

        super(DatasetHandlerHooknetCancer, self).__init__(slides_dir, contours_dir, subsets_file_path,
                                                          config.PATCH_SIZE, n_images)
        self.config = config

        self.cancer_ind = config.CANCER_IND
        self.tissue_ind = config.TISSUE_IND

        self.n_classes = 3

    def create_data_generators(self):
        n_iter_train = len(self.train_img_ids) * self.config.N_ITER_TRAIN_FACTOR
        n_iter_val = self._generate_n_iter_val(self.config.BATCH_SIZE, max(self.config.LEVELS))

        train_data_gen = self._train_generator(self.config.BATCH_SIZE, self.config.LEVELS, self.config.N_TRIALS)
        val_data_gen = self._val_generator(self.config.BATCH_SIZE, self.config.LEVELS)

        return train_data_gen, val_data_gen, n_iter_train, n_iter_val

    def _train_generator(self, batch_size, levels, n_trials):

        """yields: (x_target, x_context), y_target"""

        img_ids = self.train_img_ids
        sorted_levels = sorted(levels)
        size = (self.patch_size, self.patch_size)

        while True:
            ind = np.random.choice(len(img_ids))
            img_id = img_ids[ind]

            x1_batch = np.zeros((batch_size, self.patch_size, self.patch_size, 3))
            x2_batch = np.zeros((batch_size, self.patch_size, self.patch_size, 3))
            y1_batch = np.zeros((batch_size, self.patch_size, self.patch_size, self.n_classes))

            trials = 1
            sample_count = 0
            while sample_count < batch_size:
                if trials >= n_trials:
                    ind = np.random.choice(len(img_ids))
                    img_id = img_ids[ind]

                slide = self.slides[img_id]
                center_location = self._get_random_location(slide, size, sorted_levels)
                ret = slide.get_patches_hooknet(center_location,
                                                size,
                                                sorted_levels,
                                                True,
                                                self.config.BG_THRESHOLD)
                if ret is not None:
                    patches, seg_maps = ret

                    x1_batch[sample_count, :, :, :] = patches[0]
                    x2_batch[sample_count, :, :, :] = patches[1]
                    y1_batch[sample_count, :, :, :] = self._to_binary_maps(seg_maps[0])
                    sample_count += 1
                    trials = 1
                else:
                    trials += 1

            yield (x1_batch, x2_batch), y1_batch

    def _val_generator(self, batch_size, levels):
        img_ids = self.val_img_ids
        size = (self.patch_size, self.patch_size)

        sample_count = 0

        x1_batch = list()
        x2_batch = list()
        y1_batch = list()
        while True:
            for img_id in img_ids:

                slide = self.slides[img_id]
                dims_level0 = slide.dimensions
                downsamples = slide.level_downsamples
                ds_factor = downsamples[max(levels)]
                pad_factor = int(np.ceil(self.patch_size * ds_factor) + ds_factor)
                x_starts = range(0, dims_level0[0] - pad_factor, int(self.patch_size * ds_factor))
                y_starts = range(0, dims_level0[1] - pad_factor, int(self.patch_size * ds_factor))

                for i in x_starts:
                    for j in y_starts:

                        # location = tuple(list(
                        #     (np.array([i, j]) + np.array(size) / 2 * ds_factor).astype(
                        #         int)))

                        location = tuple(list(
                            np.ceil(np.array([i, j]) + np.array(size) / 2 * ds_factor).astype(
                                int)))

                        ret = slide.get_patches_hooknet(location,
                                                        size,
                                                        levels,
                                                        False,
                                                        self.config.VAL_DATA_BG_THRESHOLD)
                        if ret is None:
                            continue

                        patches, seg_maps = ret

                        x1_batch.append(patches[0])
                        x2_batch.append(patches[1])
                        y1_batch.append(self._to_binary_maps(seg_maps[0]))
                        sample_count += 1

                        if sample_count == batch_size:
                            sample_count = 0

                            outx1 = np.array(x1_batch)
                            outx2 = np.array(x2_batch)
                            outy1 = np.array(y1_batch)

                            x1_batch = list()
                            x2_batch = list()
                            y1_batch = list()

                            yield (outx1, outx2), outy1

    def _generate_n_iter_val(self, batch_size, highest_level):
        img_ids = self.val_img_ids
        n_patches = 0
        for img_id in img_ids:
            slide = self.slides[img_id]
            dims_level0 = slide.dimensions
            downsamples = slide.level_downsamples
            patch_size_hl = int(self.patch_size * downsamples[highest_level])
            n_w = len(
                range(0, dims_level0[0] - int(self.patch_size / 2 * downsamples[highest_level]), patch_size_hl))
            n_h = len(
                range(0, dims_level0[1] - int(self.patch_size / 2 * downsamples[highest_level]), patch_size_hl))
            n_patches += n_w * n_h
        return int(np.floor(n_patches / batch_size))

    def _to_binary_maps(self, seg_map):

        """:returns out: np.ndarray of shape(patch_size, patch_size, 3)"""

        out = np.zeros((self.patch_size, self.patch_size, 3))
        out[:, :, self.tissue_ind] = 1

        # Tumors
        for class_name in self.tumor_names:
            class_id = self.config.LABEL_DICT[class_name]
            wheres = np.where(seg_map == class_id)
            out[wheres[0], wheres[1], self.cancer_ind] = 1
            out[wheres[0], wheres[1], self.tissue_ind] = 0

        # BG
        wheres = np.where(seg_map == 0)
        out[wheres[0], wheres[1], 0] = 1
        out[wheres[0], wheres[1], self.tissue_ind] = 0

        return out


class DatasetHandlerHooknetCancerType(DatasetBase):

    def __init__(self,
                 slides_dir,
                 contours_dir,
                 subsets_file_path,
                 config=HookNetDatasetConfig,
                 n_images=None):

        super(DatasetHandlerHooknetCancerType, self).__init__(slides_dir, contours_dir, subsets_file_path,
                                                              config.PATCH_SIZE, n_images)
        self.config = config

        self.n_classes = self.config.N_CANCER + 2

    def create_data_generators(self):
        n_iter_train = len(self.train_img_ids) * self.config.N_ITER_TRAIN_FACTOR
        n_iter_val = self._generate_n_iter_val(self.config.BATCH_SIZE, max(self.config.LEVELS))

        train_data_gen = self._train_generator(self.config.BATCH_SIZE, self.config.LEVELS, self.config.N_TRIALS)
        val_data_gen = self._val_generator(self.config.BATCH_SIZE, self.config.LEVELS)

        return train_data_gen, val_data_gen, n_iter_train, n_iter_val

    def _train_generator(self, batch_size, levels, n_trials):

        """yields: (x_target, x_context), y_target"""

        img_ids = self.train_img_ids
        sorted_levels = sorted(levels)
        size = (self.patch_size, self.patch_size)

        while True:
            ind = np.random.choice(len(img_ids))
            img_id = img_ids[ind]

            x1_batch = np.zeros((batch_size, self.patch_size, self.patch_size, 3))
            x2_batch = np.zeros((batch_size, self.patch_size, self.patch_size, 3))
            y1_batch = np.zeros((batch_size, self.patch_size, self.patch_size, self.n_classes))

            trials = 1
            sample_count = 0
            while sample_count < batch_size:
                if trials >= n_trials:
                    ind = np.random.choice(len(img_ids))
                    img_id = img_ids[ind]

                slide = self.slides[img_id]
                center_location = self._get_random_location(slide, size, sorted_levels)
                ret = slide.get_patches_hooknet(center_location,
                                                size,
                                                sorted_levels,
                                                True,
                                                self.config.BG_THRESHOLD)
                if ret is not None:
                    patches, seg_maps = ret

                    x1_batch[sample_count, :, :, :] = patches[0]
                    x2_batch[sample_count, :, :, :] = patches[1]
                    y1_batch[sample_count, :, :, :] = self._to_binary_maps(seg_maps[0])
                    sample_count += 1
                    trials = 1
                else:
                    trials += 1

            yield (x1_batch, x2_batch), y1_batch

    def _val_generator(self, batch_size, levels):
        img_ids = self.val_img_ids
        size = (self.patch_size, self.patch_size)

        sample_count = 0

        x1_batch = list()
        x2_batch = list()
        y1_batch = list()
        while True:
            for img_id in img_ids:

                slide = self.slides[img_id]
                dims_level0 = slide.dimensions
                downsamples = slide.level_downsamples
                ds_factor = downsamples[max(levels)]
                pad_factor = int(np.ceil(self.patch_size * ds_factor) + ds_factor)
                x_starts = range(0, dims_level0[0] - pad_factor, int(self.patch_size * ds_factor))
                y_starts = range(0, dims_level0[1] - pad_factor, int(self.patch_size * ds_factor))

                for i in x_starts:
                    for j in y_starts:

                        # location = tuple(list(
                        #     (np.array([i, j]) + np.array(size) / 2 * ds_factor).astype(
                        #         int)))

                        location = tuple(list(
                            np.ceil(np.array([i, j]) + np.array(size) / 2 * ds_factor).astype(
                                int)))

                        ret = slide.get_patches_hooknet(location,
                                                        size,
                                                        levels,
                                                        False,
                                                        self.config.VAL_DATA_BG_THRESHOLD)
                        if ret is None:
                            continue

                        patches, seg_maps = ret

                        x1_batch.append(patches[0])
                        x2_batch.append(patches[1])
                        y1_batch.append(self._to_binary_maps(seg_maps[0]))
                        sample_count += 1

                        if sample_count == batch_size:
                            sample_count = 0

                            outx1 = np.array(x1_batch)
                            outx2 = np.array(x2_batch)
                            outy1 = np.array(y1_batch)

                            x1_batch = list()
                            x2_batch = list()
                            y1_batch = list()

                            yield (outx1, outx2), outy1

    def _generate_n_iter_val(self, batch_size, highest_level):
        img_ids = self.val_img_ids
        n_patches = 0
        for img_id in img_ids:
            slide = self.slides[img_id]
            dims_level0 = slide.dimensions
            downsamples = slide.level_downsamples
            patch_size_hl = int(self.patch_size * downsamples[highest_level])
            n_w = len(
                range(0, dims_level0[0] - int(self.patch_size / 2 * downsamples[highest_level]), patch_size_hl))
            n_h = len(
                range(0, dims_level0[1] - int(self.patch_size / 2 * downsamples[highest_level]), patch_size_hl))
            n_patches += n_w * n_h
        return int(np.floor(n_patches / batch_size))

    def _to_binary_maps(self, seg_map):

        """:returns out: np.ndarray of shape(patch_size, patch_size, n_classes)"""

        n_classes = self.config.N_CANCER + 2
        sh = seg_map.shape[:2]
        out = np.zeros(sh + (n_classes,))
        out[:, :, self.config.TISSUE_IND] = 1

        # Tumors
        for i, class_name in enumerate(self.config.TUMOR_NAMES):
            class_id = self.config.LABEL_DICT[class_name]
            wheres = np.where(seg_map == class_id)
            out[wheres[0], wheres[1], i + 1] = 1
            out[wheres[0], wheres[1], self.config.TISSUE_IND] = 0

        # BG
        wheres = np.where(seg_map == 0)
        out[wheres[0], wheres[1], 0] = 1
        out[wheres[0], wheres[1], self.config.TISSUE_IND] = 0

        return out

    def vis_sample(self, x, seg_map):

        """Visualizes the output of the data generators."""

        image1, image2 = x

        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(30, 10))

        axes[0][0].imshow(image1 / 255)
        axes[0][1].set_title('patch 1')

        axes[0][1].imshow(image2 / 255)
        axes[0][1].set_title('patch 2')

        for i, ax in enumerate(axes.flatten()[2:-1]):
            ax.imshow(seg_map[:, :, i + 1], vmin=0, vmax=1)
            cancer_type = self.config.TUMOR_NAMES[i]
            ax.set_title(f'Cancer - {cancer_type}')

        axes[-1][-1].imshow(seg_map[:, :, self.config.TISSUE_IND], vmin=0, vmax=1)
        axes[-1][-1].set_title('Tissue')


class DatasetHandlerUNetCancer(DatasetBase):

    def __init__(self,
                 slides_dir,
                 contours_dir,
                 subsets_file_path,
                 config=UNetDatasetConfig,
                 n_images=None):

        """:arg config: UNetDatasetConfig"""

        super(DatasetHandlerUNetCancer, self).__init__(slides_dir, contours_dir, subsets_file_path,
                                                       config.PATCH_SIZE, n_images)
        self.config = config

        self.cancer_ind = config.CANCER_IND
        self.tissue_ind = config.TISSUE_IND

    def create_data_generators(self,
                               out_res=None):
        n_iter_train = len(self.train_img_ids) * self.config.N_ITER_TRAIN_FACTOR
        n_iter_val = self._generate_n_iter_val(self.config.BATCH_SIZE, self.config.LEVEL)

        train_data_gen = self._train_generator(self.config.BATCH_SIZE,
                                               self.config.LEVEL,
                                               out_res,
                                               self.config.N_TRIALS)
        val_data_gen = self._val_generator(self.config.BATCH_SIZE,
                                           self.config.LEVEL, out_res)

        return train_data_gen, val_data_gen, n_iter_train, n_iter_val

    def vis_sample(self, image, seg_map):

        """Visualizes the output of the data generators."""

        fig, axes = plt.subplots(ncols=4, figsize=(10, 30))

        axes[0].imshow(image / 255)

        axes[1].imshow(seg_map[:, :, 0], vmin=0, vmax=1)
        axes[1].set_title('BG')

        axes[2].imshow(seg_map[:, :, self.cancer_ind], vmin=0, vmax=1)
        axes[2].set_title('Cancer')

        axes[3].imshow(seg_map[:, :, self.tissue_ind], vmin=0, vmax=1)
        axes[3].set_title('Tissue')

    def _to_binary_maps(self, seg_map):

        """:returns out: np.ndarray of shape(patch_size, patch_size, 3)"""

        out = np.zeros((self.patch_size, self.patch_size, 3))
        out[:, :, self.tissue_ind] = 1

        # Tumors
        for class_name in self.tumor_names:
            class_id = self.config.LABEL_DICT[class_name]
            wheres = np.where(seg_map == class_id)
            out[wheres[0], wheres[1], self.cancer_ind] = 1
            out[wheres[0], wheres[1], self.tissue_ind] = 0

        # BG
        wheres = np.where(seg_map == 0)
        out[wheres[0], wheres[1], 0] = 1
        out[wheres[0], wheres[1], self.tissue_ind] = 0

        return out

    def _generate_n_iter_val(self, batch_size, highest_level):
        img_ids = self.val_img_ids
        n_patches = 0
        for img_id in img_ids:
            slide = self.slides[img_id]
            dims_level0 = slide.dimensions
            downsamples = slide.level_downsamples
            patch_size_hl = int(self.patch_size * downsamples[highest_level])
            n_w = len(range(0, dims_level0[0] - int(self.patch_size / 2 * downsamples[highest_level]), patch_size_hl))
            n_h = len(range(0, dims_level0[1] - int(self.patch_size / 2 * downsamples[highest_level]), patch_size_hl))
            n_patches += n_w * n_h
        return int(np.floor(n_patches / batch_size))

    def _train_generator(self, batch_size, level, out_res, n_trials):
        img_ids = self.train_img_ids
        size = (self.patch_size, self.patch_size)

        while True:
            ind = np.random.choice(len(img_ids))
            img_id = img_ids[ind]

            x_batch = np.zeros((batch_size, self.patch_size, self.patch_size, 3))
            y_batch = np.zeros((batch_size, self.patch_size, self.patch_size, 3))

            trials = 1
            sample_count = 0
            while sample_count < batch_size:
                if trials >= n_trials:
                    ind = np.random.choice(len(img_ids))
                    img_id = img_ids[ind]

                slide = self.slides[img_id]
                center_location = self._get_random_location(slide, size, [level])

                patch, seg_map = slide.get_patch_unet(center_location,
                                                      size,
                                                      level,
                                                      True,
                                                      self.config.BG_THRESHOLD,
                                                      None)

                if patch is not None:
                    x_batch[sample_count, :, :, :] = patch
                    y_batch[sample_count, :, :, :] = self._to_binary_maps(seg_map)
                    sample_count += 1
                    trials = 1
                else:
                    trials += 1
            if out_res:
                y_batch = tf.image.resize(y_batch, out_res, method=self.config.RESIZE_METHOD)

            yield x_batch, y_batch

    def _val_generator(self, batch_size, level, out_res):
        img_ids = self.val_img_ids
        size = (self.patch_size, self.patch_size)

        sample_count = 0

        x_batch = list()
        y_batch = list()
        while True:
            for img_id in img_ids:

                slide = self.slides[img_id]
                dims_level0 = slide.dimensions
                downsamples = slide.level_downsamples
                ds_factor = downsamples[level]
                pad_factor = int(np.ceil(self.patch_size * ds_factor) + ds_factor)
                x_starts = range(0, dims_level0[0] - pad_factor, int(self.patch_size * ds_factor))
                y_starts = range(0, dims_level0[1] - pad_factor, int(self.patch_size * ds_factor))

                for i in x_starts:
                    for j in y_starts:

                        location = tuple(list(
                            np.ceil(np.array([i, j]) + np.array(size) / 2 * ds_factor).astype(
                                int)))

                        patch, seg_map = slide.get_patch_unet(location,
                                                              size,
                                                              level,
                                                              False,
                                                              self.config.VAL_DATA_BG_THRESHOLD,
                                                              None)
                        if patch is None:
                            continue

                        x_batch.append(patch)
                        y_batch.append(self._to_binary_maps(seg_map))
                        sample_count += 1

                        if sample_count == batch_size:
                            sample_count = 0

                            outx = np.array(x_batch)
                            if out_res:
                                outy = tf.image.resize(np.array(y_batch), out_res, method=self.config.RESIZE_METHOD)
                            else:
                                outy = np.array(y_batch)

                            x_batch = list()
                            y_batch = list()

                            yield outx, outy


class DatasetHandlerUNetCancerType(DatasetBase):

    def __init__(self,
                 slides_dir,
                 contours_dir,
                 subsets_file_path,
                 config=UNetDatasetConfig,
                 n_images=None):

        """:arg config: UNetDatasetConfig"""

        super(DatasetHandlerUNetCancerType, self).__init__(slides_dir,
                                                           contours_dir,
                                                           subsets_file_path,
                                                           config.PATCH_SIZE,
                                                           n_images)
        self.config = config

        self.bg_ind = 0
        self.tissue_ind = self.config.N_CANCER + 1
        self.n_classes = self.config.N_CANCER + 2  # Cancers + BG + Tissues

    def create_data_generators(self,
                               out_res=None):
        n_iter_train = len(self.train_img_ids) * self.config.N_ITER_TRAIN_FACTOR
        n_iter_val = self._generate_n_iter_val(self.config.BATCH_SIZE, self.config.LEVEL)

        train_data_gen = self._train_generator(self.config.BATCH_SIZE,
                                               self.config.LEVEL,
                                               out_res,
                                               self.config.N_TRIALS)
        val_data_gen = self._val_generator(self.config.BATCH_SIZE,
                                           self.config.LEVEL, out_res)

        return train_data_gen, val_data_gen, n_iter_train, n_iter_val

    def vis_sample(self, image, seg_map):

        """Visualizes the output of the data generators."""

        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(30, 10))

        axes[0][0].imshow(image / 255)

        axes[0][1].imshow(seg_map[:, :, 0], vmin=0, vmax=1)
        axes[0][1].set_title('BG')

        for i, ax in enumerate(axes.flatten()[2:-1]):
            ax.imshow(seg_map[:, :, i + 1], vmin=0, vmax=1)
            cancer_type = self.config.TUMOR_NAMES[i]
            ax.set_title(f'Cancer - {cancer_type}')

        axes[-1][-1].imshow(seg_map[:, :, self.tissue_ind], vmin=0, vmax=1)
        axes[-1][-1].set_title('Tissue')

    def _generate_n_iter_val(self, batch_size, highest_level):
        img_ids = self.val_img_ids
        n_patches = 0
        for img_id in img_ids:
            slide = self.slides[img_id]
            dims_level0 = slide.dimensions
            downsamples = slide.level_downsamples
            patch_size_hl = int(self.patch_size * downsamples[highest_level])
            n_w = len(range(0, dims_level0[0] - int(self.patch_size / 2 * downsamples[highest_level]), patch_size_hl))
            n_h = len(range(0, dims_level0[1] - int(self.patch_size / 2 * downsamples[highest_level]), patch_size_hl))
            n_patches += n_w * n_h
        return int(np.floor(n_patches / batch_size))

    def _train_generator(self, batch_size, level, out_res, n_trials):
        img_ids = self.train_img_ids
        size = (self.patch_size, self.patch_size)

        while True:
            ind = np.random.choice(len(img_ids))
            img_id = img_ids[ind]

            x_batch = np.zeros((batch_size, self.patch_size, self.patch_size, 3))
            y_batch = np.zeros((batch_size, self.patch_size, self.patch_size, self.n_classes))

            trials = 1
            sample_count = 0
            while sample_count < batch_size:
                if trials >= n_trials:
                    ind = np.random.choice(len(img_ids))
                    img_id = img_ids[ind]

                slide = self.slides[img_id]
                center_location = self._get_random_location(slide, size, [level])

                patch, seg_map = slide.get_patch_unet(center_location,
                                                      size,
                                                      level,
                                                      True,
                                                      self.config.BG_THRESHOLD,
                                                      None)

                if patch is not None:
                    x_batch[sample_count, :, :, :] = patch
                    y_batch[sample_count, :, :, :] = self._to_binary_maps(seg_map)
                    sample_count += 1
                    trials = 1
                else:
                    trials += 1
            if out_res:
                y_batch = tf.image.resize(y_batch, out_res, method=self.config.RESIZE_METHOD)

            yield x_batch, y_batch

    def _val_generator(self, batch_size, level, out_res):
        img_ids = self.val_img_ids
        size = (self.patch_size, self.patch_size)

        sample_count = 0

        x_batch = list()
        y_batch = list()
        while True:
            for img_id in img_ids:

                slide = self.slides[img_id]
                dims_level0 = slide.dimensions
                downsamples = slide.level_downsamples
                ds_factor = downsamples[level]
                pad_factor = int(np.ceil(self.patch_size * ds_factor) + ds_factor)
                x_starts = range(0, dims_level0[0] - pad_factor, int(self.patch_size * ds_factor))
                y_starts = range(0, dims_level0[1] - pad_factor, int(self.patch_size * ds_factor))

                for i in x_starts:
                    for j in y_starts:

                        location = tuple(list(
                            np.ceil(np.array([i, j]) + np.array(size) / 2 * ds_factor).astype(
                                int)))

                        patch, seg_map = slide.get_patch_unet(location,
                                                              size,
                                                              level,
                                                              False,
                                                              self.config.VAL_DATA_BG_THRESHOLD,
                                                              None)
                        if patch is None:
                            continue

                        x_batch.append(patch)
                        y_batch.append(self._to_binary_maps(seg_map))
                        sample_count += 1

                        if sample_count == batch_size:
                            sample_count = 0

                            outx = np.array(x_batch)
                            if out_res:
                                outy = tf.image.resize(np.array(y_batch), out_res, method=self.config.RESIZE_METHOD)
                            else:
                                outy = np.array(y_batch)

                            x_batch = list()
                            y_batch = list()

                            yield outx, outy

    def _to_binary_maps(self, seg_map):

        """:returns out: np.ndarray of shape(patch_size, patch_size, 3)"""

        out = np.zeros((self.patch_size, self.patch_size, self.n_classes))
        out[:, :, self.tissue_ind] = 1

        # Tumors
        for i, class_name in enumerate(self.tumor_names):
            class_id = self.config.LABEL_DICT[class_name]
            wheres = np.where(seg_map == class_id)
            out[wheres[0], wheres[1], i + 1] = 1
            out[wheres[0], wheres[1], self.tissue_ind] = 0

        # BG
        wheres = np.where(seg_map == 0)
        out[wheres[0], wheres[1], 0] = 1
        out[wheres[0], wheres[1], self.tissue_ind] = 0

        return out


class DatasetHandlerBinary:

    def __init__(self,
                 slides_dir,
                 contours_dir,
                 subsets_file_path,
                 config=UNetDatasetConfig,
                 n_images=None):

        """:arg config: UNetDatasetConfig"""

        self.config = config
        self.tumor_names = ['PNST', 'Mast Cell Tumor', 'Melanoma', 'Plasmacytoma',
                            'SCC', 'Trichoblastoma', 'Histiocytoma']
        assert config.CLASS_NAME in self.tumor_names, 'class_name must be one of {}'.format(self.tumor_names)

        self.class_name = config.CLASS_NAME
        self.target_dir = slides_dir
        self.contours_dir = contours_dir
        self.patch_size = config.PATCH_SIZE

        self.image_sets_api = None
        self.annotations_api = None
        self.annotation_types_api = None
        self.images_api = None
        self.product_api = None
        self.image_set = None
        self.images_list = None
        self.annotation_types = {}
        self._configure(n_images)

        self.df = self._load_subsets(subsets_file_path)

        print('generating slides ...')

        self.train_img_ids = list()
        self.val_img_ids = list()
        self.test_img_ids = list()
        self.slides = dict()
        self._generate_slides()
        print('{} slides generated.'.format(len(self.slides)))

        self.n_iter_val = None

    @staticmethod
    def _load_subsets(file_path):

        def data_subset(row):
            return row[0].split(';')[-1]

        def img_name(row):
            return row[0].split(';')[0]

        df = pd.read_csv(file_path)
        df['subset'] = df.apply(data_subset, axis=1)
        df['name'] = df.apply(img_name, axis=1)
        df.drop('Slide;Dataset', inplace=True, axis=1)
        return df

    def _configure(self, n_images):

        """Pass n_images=None to consider all images."""

        configuration = Configuration()
        configuration.username = 'DemoCanineSkinTumors'
        configuration.password = 'demodemo'
        configuration.host = "https://exact.cs.fau.de/"

        client = ApiClient(configuration)

        self.image_sets_api = ImageSetsApi(client)
        self.annotations_api = AnnotationsApi(client)
        self.annotation_types_api = AnnotationTypesApi(client)
        self.images_api = ImagesApi(client)
        self.product_api = ProductsApi(client)

        if self.class_name == 'PNST':
            self.image_set = self.image_sets_api.list_image_sets(name__contains='Peripheral nerve sheath tumor').results[0]
        elif self.class_name == 'SCC':
            self.image_set = self.image_sets_api.list_image_sets(name__contains='Squamous cell carcinoma').results[0]
        elif self.class_name == 'Mast Cell Tumor':
            self.image_set = self.image_sets_api.list_image_sets(name__contains='Mast cell tumor').results[0]
        else:
            self.image_set = self.image_sets_api.list_image_sets(name__contains=self.class_name).results[0]

        images_list = self.images_api.list_images(image_set=self.image_set.id, pagination=False).results
        if n_images is None:
            self.images_list = images_list
        else:
            self.images_list = images_list[:n_images]

        for product in self.image_set.product_set:
            for anno_type in self.annotation_types_api.list_annotation_types(product=product).results:
                self.annotation_types[anno_type.id] = anno_type

    def _generate_slides(self):
        for image in self.images_list:
            if image.filename not in self.df['name'].values:
                print(f'image {image.filename} not found in subsets file, skipping.')
                continue
            image_path = self.target_dir / image.name
            try:
                slide = SlideContainer(image.id,
                                       image_path,
                                       self.images_api,
                                       self.annotations_api,
                                       DatasetConfig.LABEL_DICT,
                                       self.annotation_types,
                                       self.contours_dir)
                slide.image_set_name = self.class_name
            except Exception as e:
                print('skipping {} due to {}'.format(image.name, e))
            else:
                if slide.slide.level_count < 4:
                    print('skipping {} due to absence of level 3.'.format(image.name))
                    continue

                subset = self.df[self.df['name'] == image.filename]['subset'].values[0]
                if subset == 'val':
                    self.val_img_ids.append(image.id)
                elif subset == 'test':
                    self.test_img_ids.append(image.id)
                else:
                    self.train_img_ids.append(image.id)
                self.slides[image.id] = slide

                # subset = self.df[self.df['name'] == image.filename]['subset'].values[0]
                # if subset in ('val', 'test'):
                #     self.val_img_ids.append(image.id)
                # else:
                #     self.train_img_ids.append(image.id)
                # self.slides[image.id] = slide
        print(f'{len(self.train_img_ids)} train slides, {self.val_img_ids} validation slides.')

    def _train_test_split(self, test_size=0.3):
        # assert len(self.images_list) > 3, 'Not enough images'

        img_ids = list(self.slides.keys())
        img_ids.sort()

        return train_test_split(img_ids, test_size=test_size, random_state=7)

    def _generate_n_iter_val(self, batch_size, highest_level):
        img_ids = self.val_img_ids
        n_patches = 0
        for img_id in img_ids:
            slide = self.slides[img_id]
            dims_level0 = slide.dimensions
            downsamples = slide.level_downsamples
            patch_size_hl = int(self.patch_size * downsamples[highest_level])
            n_w = len(range(0, dims_level0[0] - int(self.patch_size / 2 * downsamples[highest_level]), patch_size_hl))
            n_h = len(range(0, dims_level0[1] - int(self.patch_size / 2 * downsamples[highest_level]), patch_size_hl))
            n_patches += n_w * n_h
        self.n_iter_val = int(np.floor(n_patches / batch_size))

    def _to_binary_maps(self, seg_map):
        class_id = self.config.LABEL_DICT[self.class_name]

        out = np.zeros((self.patch_size, self.patch_size, 1))
        wheres = np.where(seg_map == class_id)
        out[wheres[0], wheres[1], 0] = 1
        return out

    @staticmethod
    def _get_random_location(slide, size, sorted_levels):

        """Returns a random location based on coordinates of level0"""

        downsample = slide.level_downsamples[sorted_levels[-1]]
        dims_level0 = slide.dimensions
        pad_x = np.ceil(size[0] / 2 * downsample).astype(int) + int(downsample)
        pad_y = np.ceil(size[1] / 2 * downsample).astype(int) + int(downsample)
        x_allowed = np.arange(pad_x, dims_level0[0] - pad_x)
        y_allowed = np.arange(pad_y, dims_level0[1] - pad_y)

        x = np.random.choice(x_allowed)
        y = np.random.choice(y_allowed)

        location = (x, y)

        return location


class UNetDataHandlerBinary(DatasetHandlerBinary):

    def create_data_generators(self,
                               out_res=None):
        train_data_gen = self._train_generator(self.config.BATCH_SIZE,
                                               self.config.LEVEL,
                                               out_res,
                                               self.config.N_TRIALS)
        self._generate_n_iter_val(self.config.BATCH_SIZE, self.config.LEVEL)
        val_data_gen = self._val_generator(self.config.BATCH_SIZE, self.config.LEVEL, out_res)
        n_iter_train = len(self.train_img_ids) * self.config.N_ITER_TRAIN_FACTOR
        return train_data_gen, val_data_gen, n_iter_train, self.n_iter_val

    def _train_generator(self, batch_size, level, out_res, n_trials):
        img_ids = self.train_img_ids
        size = (self.patch_size, self.patch_size)

        while True:
            ind = np.random.choice(len(img_ids))
            img_id = img_ids[ind]

            x_batch = np.zeros((batch_size, self.patch_size, self.patch_size, 3))
            y_batch = np.zeros((batch_size, self.patch_size, self.patch_size, 1))

            trials = 1
            sample_count = 0
            while sample_count < batch_size:
                if trials >= n_trials:
                    ind = np.random.choice(len(img_ids))
                    img_id = img_ids[ind]

                slide = self.slides[img_id]
                center_location = self._get_random_location(slide, size, [level])

                patch, seg_map = slide.get_patch_unet(center_location,
                                                      size,
                                                      level,
                                                      True,
                                                      self.config.MIN_CLASS_AREA,
                                                      self.config.LABEL_DICT[self.class_name])

                if patch is not None:
                    x_batch[sample_count, :, :, :] = patch
                    y_batch[sample_count, :, :, :] = self._to_binary_maps(seg_map)
                    sample_count += 1
                    trials = 1
                else:
                    trials += 1
            if out_res:
                y_batch = tf.image.resize(y_batch, out_res, method=self.config.RESIZE_METHOD)

            yield x_batch, y_batch

    def _val_generator(self, batch_size, level, out_res):
        img_ids = self.val_img_ids
        size = (self.patch_size, self.patch_size)

        sample_count = 0

        x_batch = list()
        y_batch = list()
        while True:
            for img_id in img_ids:

                slide = self.slides[img_id]
                dims_level0 = slide.dimensions
                downsamples = slide.level_downsamples
                ds_factor = downsamples[level]
                pad_factor = int(np.ceil(self.patch_size * ds_factor) + ds_factor)
                x_starts = range(0, dims_level0[0] - pad_factor, int(self.patch_size * ds_factor))
                y_starts = range(0, dims_level0[1] - pad_factor, int(self.patch_size * ds_factor))

                for i in x_starts:
                    for j in y_starts:

                        location = tuple(list(
                            np.ceil(np.array([i, j]) + np.array(size) / 2 * ds_factor).astype(
                                int)))

                        patch, seg_map = slide.get_patch_unet(location,
                                                              size,
                                                              level,
                                                              False,
                                                              self.config.VAL_DATA_BG_THRESHOLD)
                        if patch is None:
                            continue

                        x_batch.append(patch)
                        y_batch.append(self._to_binary_maps(seg_map))
                        sample_count += 1

                        if sample_count == batch_size:
                            sample_count = 0

                            outx = np.array(x_batch)
                            if out_res:
                                outy = tf.image.resize(np.array(y_batch), out_res, method=self.config.RESIZE_METHOD)
                            else:
                                outy = np.array(y_batch)

                            x_batch = list()
                            y_batch = list()

                            yield outx, outy


def get_subsets(file_path, class_name='Melanoma', merge_test_val=True):

    def data_subset(row):
        # print(row[0])
        return row[0].split(';')[-1]

    def img_name(row):
        return row[0].split(';')[0]

    df = pd.read_csv(file_path)
    df['subset'] = df.apply(data_subset, axis=1)
    df['name'] = df.apply(img_name, axis=1)
    df.drop('Slide;Dataset', inplace=True, axis=1)

    train_images = list()
    val_images = list()
    test_images = list()

    for i, (subset, name) in df.iterrows():
        if class_name in name or class_name.lower() in name.lower():
            if subset == 'train':
                train_images.append(name)
            elif subset == 'val':
                val_images.append(name)
            elif subset == 'test':
                test_images.append(name)
            else:
                print(f'WTF! => {subset}')
    print(f'{len(train_images)} train images')
    print(f'{len(val_images)} validation images')
    print(f'{len(test_images)} test images')

    if merge_test_val:
        val_images.extend(test_images)
        return train_images, val_images
    else:
        return train_images, val_images, test_images


def calculate_class_weights(slides_dir, contours_dir, subsets_path):

    slides_dir = Path(slides_dir)
    contours_dir = Path(contours_dir)

    dataset = DatasetBase(slides_dir=slides_dir,
                          contours_dir=contours_dir,
                          subsets_file_path=subsets_path,
                          patch_size=(256, 256))

    non_bg = list(DatasetConfig.LABEL_DICT.keys())
    non_bg.remove('Bg')
    tissue_names = [i for i in non_bg if i not in DatasetConfig.TUMOR_NAMES]
    tissue_indxs = [DatasetConfig.LABEL_DICT[i] for i in tissue_names]

    proportion_lists = {'BG': list(), 'Tissue': list()}
    for cancer_name in DatasetConfig.TUMOR_NAMES:
        proportion_lists[cancer_name] = list()

    max_dims = max([np.prod(s.dimensions) for s in dataset.slides.values()])

    skipped_slides = list()

    for slide_id, slide in dataset.slides.items():
        print('processing slide ', slide_id)

        level = 0
        size = slide.slide.level_dimensions[level]
        loc = (0, 0)
        seg_map = slide.generate_segmentation_map_for_patch(loc, size, level).flatten()
        # total_samples = np.prod(seg_map.shape)
        total_samples = len(seg_map)

        try:
            counts = np.bincount(seg_map)
        except MemoryError:
            print(f'skipping slide {slide_id} due to MemoryError ...')
            skipped_slides.append(slide_id)
        else:
            print(f'bin count: {counts}')

            w = total_samples / max_dims

            # Tissue
            tissue_proportion = 0
            for indx in tissue_indxs:
                # samples = len(np.where(seg_map == indx)[0])
                try:
                    samples = counts[indx]
                except IndexError:
                    samples = 0
                except Exception as e:
                    print(e)
                    raise e

                tissue_proportion += samples / total_samples
            proportion_lists['Tissue'].append(tissue_proportion * w)

            # Cancers
            cancer_proportion = 0
            for cancer_name in DatasetConfig.TUMOR_NAMES:
                indx = DatasetConfig.LABEL_DICT[cancer_name]
                # samples = len(np.where(seg_map == indx)[0])
                try:
                    samples = counts[indx]
                except IndexError:
                    samples = 0
                except Exception as e:
                    print(e)
                    raise e

                p = samples / total_samples
                cancer_proportion += p
                proportion_lists[cancer_name].append(p * w)

            # BG
            # bg_samples = len(np.where(seg_map == 0)[0])
            # bg_proportion = bg_samples / total_samples
            # proportion_lists['BG'].append(bg_proportion * w)
            bg_proportion = 1 - (tissue_proportion + cancer_proportion)
            proportion_lists['BG'].append(bg_proportion * w)

    print(f'skipped slides: {skipped_slides}')

    sums = {k: sum(v) for k, v in proportion_lists.items()}
    total_sums = sum(list(sums.values()))

    proportions = {k: v / total_sums for k, v in sums.items()}
    max_p = max(list(proportions.values()))
    weights = {k: max_p / v for k, v in proportions.items()}
    print('Done.')
    print(weights)
    return weights
