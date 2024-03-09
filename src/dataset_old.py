import numpy as np

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
from .config import DatasetConfig, ONLY_MELANOMA, UNetDatasetConfig


class DatasetHandler:

    """Handling the data preparation.

    Attributes:
        target_dir (path): directory to save images.
        annotation_types (dict): a dictionary to map from available 'annotation_id's to 'annotation_name's
        images_list (list): a list of image ids. Each image is accessible in this way as a OpenSlide:
            >>> slide = (self).slides[image_id]
        label_dict (dict): a dictionary mapping from label names to label ids
    """

    def __init__(self,
                 slides_dir,
                 contours_dir,
                 patch_size=DatasetConfig.PATCH_SIZE,
                 n_images=None):
        # if not os.path.isdir(target_dir):
        #     os.mkdir(target_dir)
        self.target_dir = slides_dir
        self.contours_dir = contours_dir
        self.patch_size = patch_size

        self.image_sets_api = None
        self.annotations_api = None
        self.annotation_types_api = None
        self.images_api = None
        self.product_api = None
        self.image_set = None
        self.images_list = None
        self.annotation_types = {}
        self._configure(n_images)

        self.label_dict = DatasetConfig.LABEL_DICT
        self.n_classes = DatasetConfig.N_CLASSES
        print('number of classes: ', self.n_classes)

        print('generating slides ...')
        self.slides = dict()
        self._generate_slides()
        print('{} slides generated.'.format(len(self.slides)))

        (self.train_img_ids, self.val_img_ids) = self._train_test_split(test_size=DatasetConfig.TEST_SIZE)
        self.n_iter_val = None

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
        out = np.zeros((self.patch_size, self.patch_size, self.n_classes))
        for i in range(self.n_classes):
            wheres = np.where(seg_map == i)
            out[wheres[0], wheres[1], i] = 1
        return out

    def _train_test_split(self, test_size=0.3):
        # assert len(self.images_list) > 3, 'Not enough images'

        img_ids = list(self.slides.keys())
        img_ids.sort()

        return train_test_split(img_ids, test_size=test_size, random_state=7)

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

        if ONLY_MELANOMA:
            self.image_set = self.image_sets_api.list_image_sets(name__contains='Melanoma').results[0]

            images_list = self.images_api.list_images(image_set=self.image_set.id, pagination=False).results
            if n_images is None:
                self.images_list = images_list
            else:
                self.images_list = images_list[:n_images]

            for product in self.image_set.product_set:
                for anno_type in self.annotation_types_api.list_annotation_types(product=product).results:
                    self.annotation_types[anno_type.id] = anno_type

        else:
            image_sets = self.image_sets_api.list_image_sets().results
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
            image_path = self.target_dir / image.name
            try:
                slide = SlideContainer(image.id,
                                       image_path,
                                       self.images_api,
                                       self.annotations_api,
                                       self.label_dict,
                                       self.annotation_types,
                                       self.contours_dir)
            except Exception as e:
                print('skipping {} due to {}'.format(image.name, e))
            else:
                if slide.slide.level_count < 4:
                    print('skipping {} due to absence of level 3.'.format(image.name))
                    continue
                self.slides[image.id] = slide


class HookNetDataHandler(DatasetHandler):

    def create_data_generators(self, batch_size=DatasetConfig.BATCH_SIZE, levels=DatasetConfig.LEVELS):
        train_data_gen = self._train_generator(batch_size, levels)
        self._generate_n_iter_val(batch_size, max(levels))
        val_data_gen = self._val_generator(batch_size, levels)
        n_iter_train = len(self.images_list) * 5
        return train_data_gen, val_data_gen, n_iter_train, self.n_iter_val

    def _train_generator(self, batch_size, levels):

        """"""

        img_ids = self.train_img_ids
        sorted_levels = sorted(levels)
        size = (self.patch_size, self.patch_size)

        while True:
            ind = np.random.choice(len(img_ids))
            img_id = img_ids[ind]

            x1_batch = np.zeros((batch_size, self.patch_size, self.patch_size, 3))
            x2_batch = np.zeros((batch_size, self.patch_size, self.patch_size, 3))
            y1_batch = np.zeros((batch_size, self.patch_size, self.patch_size, self.n_classes))
            y2_batch = np.zeros((batch_size, self.patch_size, self.patch_size, self.n_classes))

            sample_count = 0
            while sample_count < batch_size:
                slide = self.slides[img_id]
                center_location = self._get_random_location(slide, size, sorted_levels)
                patches, seg_maps = slide.get_patches_hooknet(center_location,
                                                              size,
                                                              sorted_levels,
                                                              True,
                                                              DatasetConfig.BG_THRESHOLD)

                if patches is not None:
                    x1_batch[sample_count, :, :, :] = patches[0]
                    x2_batch[sample_count, :, :, :] = patches[1]
                    y1_batch[sample_count, :, :, :] = self._to_binary_maps(seg_maps[0])
                    y2_batch[sample_count, :, :, :] = self._to_binary_maps(seg_maps[1])
                    sample_count += 1

            yield (x1_batch, x2_batch), (y1_batch, y2_batch)

    def _val_generator(self, batch_size, levels):
        img_ids = self.val_img_ids
        size = (self.patch_size, self.patch_size)

        sample_count = 0

        x1_batch = list()
        x2_batch = list()
        y1_batch = list()
        y2_batch = list()
        while True:
            for img_id in img_ids:

                slide = self.slides[img_id]
                dims_level0 = slide.dimensions
                downsamples = slide.level_downsamples
                ds_factor = downsamples[max(levels)]
                pad_factor = np.ceil(self.patch_size * ds_factor).astype(int)
                x_starts = range(0, dims_level0[0] - pad_factor, int(self.patch_size * ds_factor))
                y_starts = range(0, dims_level0[1] - pad_factor, int(self.patch_size * ds_factor))

                for i in x_starts:
                    for j in y_starts:

                        location = tuple(list(
                            (np.array([i, j]) + np.array(size) / 2 * ds_factor).astype(
                                int)))

                        patches, seg_maps = slide.get_patches_hooknet(location,
                                                                      size,
                                                                      levels,
                                                                      False,
                                                                      0.5)
                        if patches is None:
                            continue

                        x1_batch.append(patches[0])
                        x2_batch.append(patches[1])
                        y1_batch.append(self._to_binary_maps(seg_maps[0]))
                        y2_batch.append(self._to_binary_maps(seg_maps[1]))
                        sample_count += 1

                        if sample_count == batch_size:
                            sample_count = 0

                            outx1 = np.array(x1_batch)
                            outx2 = np.array(x2_batch)
                            outy1 = np.array(y1_batch)
                            outy2 = np.array(y2_batch)

                            x1_batch = list()
                            x2_batch = list()
                            y1_batch = list()
                            y2_batch = list()

                            yield (outx1, outx2), (outy1, outy2)


class UNetDataHandler(DatasetHandler):

    def create_data_generators(self, out_res=None, batch_size=DatasetConfig.BATCH_SIZE, level=UNetDatasetConfig.LEVEL):
        train_data_gen = self._train_generator(batch_size, level, out_res)
        self._generate_n_iter_val(batch_size, level)
        val_data_gen = self._val_generator(batch_size, level, out_res)
        n_iter_train = len(self.slides) * 10
        return train_data_gen, val_data_gen, n_iter_train, self.n_iter_val

    def _train_generator(self, batch_size, level, out_res, n_trials=20):
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
                                                      DatasetConfig.BG_THRESHOLD)

                if patch is not None:
                    x_batch[sample_count, :, :, :] = patch
                    y_batch[sample_count, :, :, :] = self._to_binary_maps(seg_map)
                    sample_count += 1
                    trials = 1
                else:
                    trials += 1
            if out_res:
                y_batch = tf.image.resize(y_batch, out_res, method=DatasetConfig.RESIZE_METHOD)

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
                                                              DatasetConfig.VAL_DATA_BG_THRESHOLD)
                        if patch is None:
                            continue

                        x_batch.append(patch)
                        y_batch.append(self._to_binary_maps(seg_map))
                        sample_count += 1

                        if sample_count == batch_size:
                            sample_count = 0

                            outx = np.array(x_batch)
                            if out_res:
                                outy = tf.image.resize(np.array(y_batch), out_res, method=DatasetConfig.RESIZE_METHOD)
                            else:
                                outy = np.array(y_batch)

                            x_batch = list()
                            y_batch = list()

                            yield outx, outy