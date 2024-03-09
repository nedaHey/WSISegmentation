import argparse
import os
from pathlib import Path

from src.models import ResUNet
from src.dataset import DatasetHandlerUNetCancer
from src.config import UNetDatasetConfig, ResUnetConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir',
                        type=str,
                        help='directory of the slides',
                        required=True)
    parser.add_argument('--subset_file_path',
                        type=str,
                        help='path to subset file datasets.xls',
                        required=True)
    parser.add_argument('--out_dir',
                        type=str,
                        help='directory for saving checkpoints and logs',
                        required=True)

    parser.add_argument('--model_name',
                        default='res_unet',
                        type=str,
                        help='model name',
                        required=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    config = UNetDatasetConfig
    model_config = ResUnetConfig
    assert model_config.OUTPUT_TYPE == 'cancer', "change OUTPUT_TYPE to 'cancer' in ResUnetConfig, config.py"

    dataset_dir = Path(args.dataset_dir)
    slides_dir = dataset_dir / 'slides'
    contours_dir = dataset_dir / 'contours'

    if not slides_dir.is_dir():
        print('dataset does not exist.')
    else:
        dataset = DatasetHandlerUNetCancer(slides_dir,
                                           contours_dir,
                                           str(Path(args.subset_file_path)),
                                           config)
        train_data_gen, val_data_gen, n_iter_train, n_iter_val = dataset.create_data_generators()

        checkpoints_dir = Path(os.path.join(args.out_dir, 'checkpoints'))
        logs_dir = Path(os.path.join(args.out_dir, 'logs'))

        res_unet = ResUNet(input_shape=(config.PATCH_SIZE, config.PATCH_SIZE, 3),
                           checkpoints_dir=checkpoints_dir,
                           log_dir=logs_dir,
                           name=args.model_name,
                           config=model_config)

        res_unet.train(train_data_gen, n_iter_train, val_data_gen, n_iter_val, config.EPOCHS)
