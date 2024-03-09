import pickle

import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A

import openslide

# from .dataset import DatasetBase

try:
    import staintools
except ModuleNotFoundError as e:
    print('install staintools from here: https://hackmd.io/@peter554/staintools')

from .config import AugmentationConfig, UNetDatasetConfig, DatasetConfig


class SlideContainer:

    def __init__(self,
                 image_id,
                 image_path,
                 images_api,
                 anotations_api,
                 label_dict,
                 annotation_types,
                 contours_dir):
        self.image_id = image_id
        self.contours_dir = contours_dir
        self.image_path = image_path
        if not image_path.is_file():
            print('downloading image {} ...'.format(image_path))
            images_api.download_image(id=image_id, target_path=image_path)
        else:
            print('image {} exists.'.format(image_path))
        self.slide = openslide.OpenSlide(str(image_path))
        self.dimensions = self.slide.dimensions
        self.level_downsamples = self.slide.level_downsamples
        self.level_dimensions = self.slide.level_dimensions

        self.contours = list()
        self.labels = list()
        self.label_names = list()
        self.areas = list()
        self.label_proportions = dict()
        self._generate_contours(anotations_api, label_dict, annotation_types)
        self.image_set_name = None

        self.augment = A.Compose([A.Rotate(limit=AugmentationConfig.ROTATE_LIMIT,
                                           p=AugmentationConfig.ROTATE_P),
                                 A.RandomBrightnessContrast(p=AugmentationConfig.RANDOM_BC_P),
                                 A.RandomGamma(p=AugmentationConfig.RANDOM_GAMMA_P),
                                 A.GaussNoise(p=AugmentationConfig.GAUSS_NOISE_P)],
                                 additional_targets={"image1": "image", "mask1": "mask"})

        self.augment_unet = A.Compose([A.Rotate(limit=AugmentationConfig.ROTATE_LIMIT,
                                                p=AugmentationConfig.ROTATE_P),
                                      A.RandomBrightnessContrast(p=AugmentationConfig.RANDOM_BC_P),
                                      A.RandomGamma(p=AugmentationConfig.RANDOM_GAMMA_P),
                                      A.GaussNoise(p=AugmentationConfig.GAUSS_NOISE_P)])
        self.do_stain_aug = UNetDatasetConfig.DO_STAIN_AUG
        self.stain_sigma1 = 0.2
        self.stain_sigma2 = 0.2

    def get_patches_hooknet(self, center_location, size, sorted_levels, do_augmentation, bg_threshold=0.9):

        """
        location: center location at level0

        returns:
        [out_patches], [out_seg_maps]

        Note: Returns `None` for out_patches if the bg_area > bg_threshold == True for the target level, i.e. the
         smaller level
        """

        # self._check_limits(center_location, size, sorted_levels)

        patches = list()
        seg_maps = list()

        out_patches = np.zeros((len(sorted_levels), size[0], size[1], 3))
        out_seg_maps = np.zeros((len(sorted_levels), size[0], size[1]))

        ret = None

        # Target level
        level = sorted_levels[0]
        patch, seg_map = self.get_patch(center_location, size, level)
        bg_area = (len(np.where(seg_map == 0)[0])) / (np.prod(size))

        if bg_area <= bg_threshold:
            patches.append(patch)
            seg_maps.append(seg_map)

            # Context level
            level = sorted_levels[1]
            patch, seg_map = self.get_patch(center_location, size, level)
            patches.append(patch)
            seg_maps.append(seg_map)

            if do_augmentation:
                transformed = self.augment(image=patches[0],
                                           image1=patches[1],
                                           mask=seg_maps[0],
                                           mask1=seg_maps[1])

                out_patches[0] = transformed['image']
                out_patches[1] = transformed['image1']
                out_seg_maps[0] = transformed['mask']
                out_seg_maps[1] = transformed['mask1']
            else:
                out_patches[0] = patches[0]
                out_patches[1] = patches[1]
                out_seg_maps[0] = seg_maps[0]
                out_seg_maps[1] = seg_maps[1]

            ret = out_patches, out_seg_maps

        return ret

    def get_patch_unet(self, center_location, size, level, do_augmentation, threshold=0.9, class_id=None):

        """Threshold checking defaults to background, i.e. if class_id is None, then returns None if the patch has
        background area greater than threshold, else returns None if the patch has class_area smaller than threshold."""
        # self._check_limits(center_location, size, [level])

        out_patch = np.zeros((size[0], size[1], 3))
        out_seg_map = np.zeros((size[0], size[1]))

        patch, seg_map = self.get_patch(center_location, size, level)

        if class_id is None:
            bg_area = (len(np.where(seg_map == 0)[0])) / (np.prod(size))
            if bg_area > threshold:
                return None, None
        else:
            valid_area = (len(np.where(seg_map == class_id)[0])) / (np.prod(size))
            if valid_area < threshold:
                return None, None

        if do_augmentation:
            transformed = self.augment_unet(image=patch,
                                            mask=seg_map)

            augmented_img = transformed['image']
            augmented_mask = transformed['mask']

            if self.do_stain_aug:
                to_augment = staintools.LuminosityStandardizer.standardize(augmented_img)
                augmentor = staintools.StainAugmentor(method='vahadane',
                                                      sigma1=self.stain_sigma1,
                                                      sigma2=self.stain_sigma2)
                augmentor.fit(to_augment)
                augmented_img = augmentor.pop()

            out_patch[:, :, :] = augmented_img
            out_seg_map[:, :] = augmented_mask
        else:
            out_patch[:, :, :] = patch
            out_seg_map[:, :] = seg_map

        return out_patch, out_seg_map

    def get_patch(self, center_location, size, level, seg_map_type='default'):

        """returns the patch with `size` in `level` from `center_location` which `center_location` is in level0.

        :argument size: (width, height)

        :returns patch (size, 3), seg_map (size)(not_binary)"""

        assert seg_map_type in ('default', 'cancer', 'cancer_type')

        topleft_location = tuple(
            np.ceil(np.array(center_location) - (np.array(size) * self.level_downsamples[level] / 2)).astype(int))
        try:
            patch = np.array(self.slide.read_region(location=topleft_location,
                                                    level=level,
                                                    size=size))
        except Exception as e:
            print('exception when extracting patch from slide {}. error: '.format(self.image_path), e)
            raise e

        if patch.ndim != 3:
            raise Exception('patch ndim is not 3: {} for slide {}'.format(patch.ndim, self.image_path))

        if patch.shape[-1] < 3:
            raise Exception('patch ndim is smaller than 3: {} for slide {}'.format(patch.shape, self.image_path))

        patch = patch[:, :, 0: 3]

        seg_map = self.generate_segmentation_map_for_patch(topleft_location, size, level)

        if seg_map_type == 'cancer':
            seg_map = self.to_binary_maps_cancer(seg_map)
        elif seg_map_type == 'cancer_type':
            seg_map = self.to_binary_maps_cancer_type(seg_map)

        return patch, seg_map

    def generate_segmentation_map_for_patch(self, topleft_location, size, level):
        w, h = size
        y_patch = np.zeros(shape=(h, w), dtype=np.int8)
        downsample_factor = self.level_downsamples[level]
        coordinates = np.copy(self.contours) / downsample_factor

        for label, poly in zip(self.labels, coordinates):
            poly = poly - np.array(topleft_location) / downsample_factor
            cv2.drawContours(y_patch, [poly.reshape((-1, 1, 2)).astype(int)], -1, label, -1)

        return y_patch

    def _check_limits(self, location, size, sorted_levels):
        max_level = sorted_levels[-1]
        downsample_factor = self.level_downsamples[max_level]
        max_level_point = np.floor(np.array(location) / downsample_factor).astype(int)
        max_level_dims = self.level_dimensions[max_level]
        assert max_level_point[0] + size[0] // 2 < max_level_dims[0], 'max_level x exceeded'
        assert max_level_point[1] + size[1] // 2 < max_level_dims[1], 'max_level y exceeded'

    def _generate_contours(self, annotations_api, label_dict, annotation_types):
        contours_path = self.contours_dir / '{}.pkl'.format(self.image_id)
        if contours_path.is_file():
            pkl = self._load_pickle()
            if pkl.get('label_names') is not None:
                print('annotation data exists.')
                self.contours = pkl['contours']
                self.labels = pkl['labels']
                self.areas = pkl['areas']
                self.label_names = pkl['label_names']
                self.label_proportions = pkl['label_proportions']
                return

        print('generating annotation data ...')
        for annotation in annotations_api.list_annotations(image=self.image_id, pagination=False).results:
            vector = []
            for i in range(1, (len(annotation.vector) // 2) + 1):
                vector.append([annotation.vector['x' + str(i)], annotation.vector['y' + str(i)]])
            vector_array = np.array(vector)

            self.contours.append(vector_array)
            area = cv2.contourArea(vector_array)
            self.areas.append(area)
            name = annotation_types[annotation.annotation_type].name
            self.label_names.append(name)
            self.labels.append(label_dict[name])

        unique_labels = list(set(self.labels))
        numpy_labels = np.array(self.labels)
        numpy_areas = np.array(self.areas)
        for label in unique_labels:
            area_sum = np.sum(numpy_areas[np.where(numpy_labels == label)])
            self.label_proportions[label] = area_sum

        self._write_pickle()

    def _load_pickle(self):
        with open('{}.pkl'.format(self.contours_dir / str(self.image_id)), 'rb') as f:
            p = pickle.load(f)
        return p

    def _write_pickle(self):
        pkl = {'contours': self.contours, 'labels': self.labels,
               'areas': self.areas, 'label_proportions': self.label_proportions,
               'label_names': self.label_names}
        with open('{}.pkl'.format(self.contours_dir / str(self.image_id)), 'wb') as f:
            pickle.dump(pkl, f)

    @staticmethod
    def to_binary_maps_cancer(seg_map, ds_config=DatasetConfig):

        """:returns out: np.ndarray of shape(patch_size, patch_size, 3)"""

        sh = seg_map.shape[:2]
        out = np.zeros(sh + (3,))
        # print(sh, out.shape)
        out[:, :, ds_config.TISSUE_IND] = 1

        # Tumors
        for class_name in ds_config.TUMOR_NAMES:
            class_id = ds_config.LABEL_DICT[class_name]
            wheres = np.where(seg_map == class_id)
            out[wheres[0], wheres[1], ds_config.CANCER_IND] = 1
            out[wheres[0], wheres[1], ds_config.TISSUE_IND] = 0

        # BG
        wheres = np.where(seg_map == 0)
        out[wheres[0], wheres[1], 0] = 1
        out[wheres[0], wheres[1], ds_config.TISSUE_IND] = 0

        return out

    @staticmethod
    def to_binary_maps_cancer_type(seg_map, ds_config=DatasetConfig):

        """:returns out: np.ndarray of shape(patch_size, patch_size, 9)"""

        n_classes = ds_config.N_CANCER + 2
        sh = seg_map.shape[:2]
        out = np.zeros(sh + (n_classes, ))
        out[:, :, ds_config.TISSUE_IND] = 1

        # Tumors
        for i, class_name in enumerate(ds_config.TUMOR_NAMES):
            class_id = ds_config.LABEL_DICT[class_name]
            wheres = np.where(seg_map == class_id)
            out[wheres[0], wheres[1], i + 1] = 1
            out[wheres[0], wheres[1], ds_config.TISSUE_IND] = 0

        # BG
        wheres = np.where(seg_map == 0)
        out[wheres[0], wheres[1], 0] = 1
        out[wheres[0], wheres[1], ds_config.TISSUE_IND] = 0

        return out

    @staticmethod
    def to_binary_maps_cancer_type_v2(seg_map, ds_config=DatasetConfig):

        """:returns out: np.ndarray of shape(patch_size, patch_size)"""

        sh = seg_map.shape[:2]
        out = np.ones(sh) * (len(ds_config.TUMOR_NAMES) + 1)

        # Tumors
        for i, class_name in enumerate(ds_config.TUMOR_NAMES):
            class_id = ds_config.LABEL_DICT[class_name]
            wheres = np.where(seg_map == class_id)
            out[wheres] = i + 1
            # out[wheres[0], wheres[1], i + 1] = 1
            # out[wheres[0], wheres[1], ds_config.TISSUE_IND] = 0

        # BG
        wheres = np.where(seg_map == 0)
        out[wheres[0], wheres[1]] = 0

        return out


def visualize_patches(patches, seg_maps):
    n_patches = len(patches)
    fig, axes = plt.subplots(ncols=n_patches, nrows=n_patches, figsize=(12, 12))

    for i in range(n_patches):
        patch = patches[i]
        seg_map = seg_maps[i]

        axes[i, 0].imshow(patch)
        axes[i, 1].imshow(seg_map)
