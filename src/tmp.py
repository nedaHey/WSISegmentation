import numpy as np
from pathlib import Path
import openslide
import cv2
import random

from random import randint
from tqdm import tqdm

from exact_sync.v1.api.annotations_api import AnnotationsApi
from exact_sync.v1.api.images_api import ImagesApi
from exact_sync.v1.api.image_sets_api import ImageSetsApi
from exact_sync.v1.api.annotation_types_api import AnnotationTypesApi

from exact_sync.v1.configuration import Configuration
from exact_sync.v1.api_client import ApiClient

from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.data_block import *
from fastai.vision.data import SegmentationProcessor


class DatasetConfig:

    BG_THRESHOLD = 0.5
    PATCH_SIZE = 284

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

    N_CLASSES = len(LABEL_DICT)
    TEST_SIZE = 0.3
    BATCH_SIZE = 16
    LEVELS = (1, 3)
    UNET_INPUT_LEVEL = max(LEVELS)
    VAL_DATA_BG_THRESHOLD = 1.0

    RESIZE_METHOD = 'lanczos5'


PreProcessors = Union[PreProcessor, Collection[PreProcessor]]
fastai_types[PreProcessors] = 'PreProcessors'


def data_generator(data_bunch, is_train, class_name='Melanoma'):
    # if is_train:
    #     data_gen = iter(data_bunch.dl(DatasetType.Train))
    #     # img_batch, target_batch = data_bunch.one_batch(DatasetType.Train)
    # else:
    #     data_gen = iter(data_bunch.dl(DatasetType.Valid))
    #     # img_batch, target_batch = data_bunch.one_batch(DatasetType.Valid)
    while True:
        if is_train:
            data_gen = data_bunch.train_dl
        else:
            data_gen = data_bunch.val_dl
        for x, y in data_gen:
        # x, y = next(data_gen)
            x = data_bunch.denorm(x)
            images = x.numpy().transpose(0, 2, 3, 1) * 255
            y_b = y.numpy().transpose(0, 2, 3, 1)
            masks = np.zeros_like(y_b)
            masks[np.where(y_b == DatasetConfig.LABEL_DICT[class_name])] = 1
            yield images, masks


def get_data_bunch(target_folder,
                   train_image_names,
                   val_image_names,
                   patch_size=256,
                   levels=[0, 1, 2],
                   batch_size=8,
                   class_name='Melanoma'):
    # patch_size = 256
    # levels = [0, 1, 2]
    # target_folder = Path("/content/drive/MyDrive/dataset/melanoma")

    configuration = Configuration()
    configuration.username = 'DemoCanineSkinTumors'
    configuration.password = 'demodemo'
    configuration.host = "https://exact.cs.fau.de/"

    client = ApiClient(configuration)

    image_sets_api = ImageSetsApi(client)
    annotations_api = AnnotationsApi(client)
    annotation_types_api = AnnotationTypesApi(client)
    images_api = ImagesApi(client)

    image_sets = image_sets_api.list_image_sets(name__contains=class_name)
    container, annotation_types = generate_container(image_sets,
                                                     annotation_types_api,
                                                     images_api,
                                                     target_folder,
                                                     annotations_api,
                                                     patch_size, levels)

    # find a good train test split!
    train_slides = list()
    val_slides = list()
    for c in container:
        if c.file.name in train_image_names:
            train_slides.append(c)
        elif c.file.name in val_image_names:
            val_slides.append(c)
        else:
            print(f'WTF! => {c.file.name}')
    print('train images: ', len(train_slides))
    print('validation images: ', len(val_slides))
    # train_files = list(np.random.choice(container[:5], train_images))
    # valid_files = list(np.random.choice(container[5:], val_images))

    tfms = get_transforms(do_flip=True,
                          flip_vert=True,
                          max_rotate=45,
                          max_lighting=0.15,
                          max_zoom=2
                          )

    valid = SlideSegmentationItemList(val_slides)
    train = SlideSegmentationItemList(train_slides)

    path = Path('.')
    item_list = ItemLists(path, train, valid)
    item_list = item_list.label_from_func(lambda x: x,
                                          classes=list(DatasetConfig.LABEL_DICT.keys()),
                                          label_cls=SlideSegmentationLabelList)  # classes=['Bg', 'Vesel'],
    data = item_list.transform(tfms, size=patch_size, tfm_y=True)
    data = data.databunch(bs=batch_size)  # , num_workers=0
    data = data.normalize()
    return data


def generate_container(image_sets,
                       annotation_types_api,
                       images_api,
                       target_folder,
                       annotations_api,
                       patch_size,
                       levels):
    label_dict = DatasetConfig.LABEL_DICT
    annotation_types = {}
    container = []
    for image_set in image_sets.results:

        for product in image_set.product_set:
            for anno_type in annotation_types_api.list_annotation_types(product=product).results:
                annotation_types[anno_type.id] = anno_type

        for image in tqdm(images_api.list_images(image_set=image_set.id, pagination=False).results):
            coordinates, labels = [], []

            image_path = target_folder / image.name

            for annotation in annotations_api.list_annotations(image=image.id, pagination=False).results:
                annotation.annotation_type = annotation_types[annotation.annotation_type]

                # if file not exists download it
                if image_path.is_file() == False:
                    images_api.download_image(id=image.id, target_path=image_path)

                vector = []
                for i in range(1, (len(annotation.vector) // 2) + 1):
                    vector.append([annotation.vector['x' + str(i)], annotation.vector['y' + str(i)]])

                coordinates.append(np.array(vector))
                labels.append(label_dict[annotation.annotation_type.name])

            if len(coordinates) > 0:
                try:
                    container.append(SlideContainer(image_path, (labels, coordinates), 0, patch_size, patch_size, level_range=levels))
                except Exception as e:
                    print(e)
                    print('skipping {}'.format(image_path))

    return container, annotation_types


class SlideContainer:

    def __init__(self,
                 file: Path,
                 y,
                 level: int = 0,
                 width: int = 256,
                 height: int = 256,
                 level_range: list = [0, 1, 2, 3],
                 sample_func: callable = None):

        self.labels, self.coordinates = y
        self.file = file
        self.slide = openslide.open_slide(str(file))

        self.width = width
        self.height = height
        self.down_factor = self.slide.level_downsamples[level]

        if level is None:
            level = self.slide.level_count - 1
        self._level = level
        # remove levels that are not supported!
        self.level_range = [l for l in level_range if l < self.slide.level_count]

        self.sample_func = sample_func

    @property
    def level(self):
        return self.level

    @level.setter
    def level(self, value):
        self.down_factor = self.slide.level_downsamples[value]
        self._level = value

    @property
    def shape(self):
        return self.width, self.height

    @property
    def slide_shape(self):
        return self.slide.level_dimensions[self._level]

    def get_new_level(self):
        return random.choice(self.level_range)

    def get_patch(self, x: int = 0, y: int = 0):
        return np.array(self.slide.read_region(location=(int(x * self.down_factor), int(y * self.down_factor)),
                                               level=self._level, size=(self.width, self.height)))[:, :, :3]

    def get_y_patch(self, x: int = 0, y: int = 0):
        y_patch = np.zeros(shape=(self.width, self.height), dtype=np.int8)

        coordinates = np.copy(self.coordinates) / self.down_factor

        for label, poly in zip(self.labels, coordinates):
            poly = poly - (x, y)
            cv2.drawContours(y_patch, [poly.reshape((-1, 1, 2)).astype(int)], -1, label, -1)

        return y_patch

    def get_new_train_coordinates(self):
        # use passed sampling method
        if callable(self.sample_func):
            return self.sample_func(self.y, **{"classes": self.classes, "size": self.shape,
                                               "level_dimensions": self.slide.level_dimensions,
                                               "level": self.level})

        # use default sampling method
        label = np.random.choice(list(set(self.labels)), 1)[0]
        labels_with_coordinates = np.array(list(zip(self.labels, self.coordinates)))
        # filter by label
        labels_with_coordinates = labels_with_coordinates[labels_with_coordinates[:, 0] == label]

        poly = np.random.choice(labels_with_coordinates[:, 1], 1)[0]
        xmin, ymin = poly[np.random.choice(len(poly), 1)[0]]

        xmin, ymin = xmin / self.down_factor, ymin / self.down_factor

        xmin = max(0, int(xmin + randint(-self.width / 2, self.width / 2)))
        xmin = min(xmin, self.slide_shape[0] - self.width)

        ymin = max(0, int(ymin + randint(-self.height / 2, self.height / 2)))
        ymin = min(ymin, self.slide_shape[1] - self.height)

        return xmin, ymin

    def __str__(self):
        return str(self.path)


class SlideItemList(ItemList):

    def __init__(self, items:Iterator, path:PathOrStr='.', label_cls:Callable=None, inner_df:Any=None,
                 processor:PreProcessors=None, x:'ItemList'=None, ignore_empty:bool=False):
        self.path = Path(path)
        self.num_parts = len(self.path.parts)
        self.items,self.x,self.ignore_empty = items,x,ignore_empty
        self.sizes = [None] * len(self.items)
        if not isinstance(self.items,np.ndarray): self.items = array(self.items, dtype=object)
        self.label_cls,self.inner_df,self.processor = ifnone(label_cls,self._label_cls),inner_df,processor
        self._label_list,self._split = SlideLabelList,ItemLists
        self.copy_new = ['x', 'label_cls', 'path']

    def __getitem__(self,idxs: int, x: int=0, y: int=0)->Any:
        idxs = try_int(idxs)
        if isinstance(idxs, numbers.Integral):
            return self.get(idxs, x, y)
        else:
            return self.get(*idxs)


class SlideImageItemList(SlideItemList):
    pass


class SlideSegmentationItemList(SlideImageItemList, ImageList):

    def get(self, i, x: int, y: int):
        fn = self.items[i]
        res = self.open(fn, x, y)
        self.sizes[i] = res.size
        return res

    def open(self, fn: SlideContainer,  x: int=0, y: int=0):
        patch = fn.get_patch(x, y) / 255.

        return Image(pil2tensor(patch, np.float32))


class SlideSegmentationLabelList(ImageList, SlideImageItemList):

    """`ItemList` for segmentation masks."""

    _processor = SegmentationProcessor

    def __init__(self, items: Iterator, classes: Collection=None, **kwargs):
        super().__init__(items, **kwargs)
        self.copy_new.append('classes')
        self.classes, self.loss_func = classes, CrossEntropyFlat(axis=1)

    def get(self, i, x: int, y: int):
        fn = self.items[i]
        res = self.open(fn, x, y)
        self.sizes[i] = res.size
        return res

    def open(self, fn: SlideContainer,  x: int=0, y: int=0):
        patch = fn.get_y_patch(x, y)
        return ImageSegment(pil2tensor(patch, np.float32))

    def analyze_pred(self, pred, thresh:float=0.5): return pred.argmax(dim=0)[None]

    def reconstruct(self, t:Tensor):
        return ImageSegment(t)


class SlideLabelList(LabelList):
    def __getitem__(self, idxs: Union[int, np.ndarray]) -> 'LabelList':
        idxs = try_int(idxs)
        if isinstance(idxs, numbers.Integral):
            if self.item is None:
                slide_container = self.x.items[idxs]
                slide_container_y = self.y.items[idxs]

                level = slide_container.get_new_level()
                slide_container.level = level
                slide_container_y.level = level

                xmin, ymin = slide_container.get_new_train_coordinates()

                x = self.x.get(idxs, xmin, ymin)
                y = self.y.get(idxs, xmin, ymin)
            else:
                x, y = self.item, 0
            if self.tfms or self.tfmargs:
                x = x.apply_tfms(self.tfms, **self.tfmargs)
            if hasattr(self, 'tfms_y') and self.tfm_y and self.item is None:
                y = y.apply_tfms(self.tfms_y, **{**self.tfmargs_y, 'do_resolve': False})
            if y is None: y = 0
            return x, y
        else:
            return self.new(self.x[idxs], self.y[idxs])
