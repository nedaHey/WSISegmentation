import argparse
import os
from pathlib import Path

from src.models import HookNet
from src.dataset import DatasetHandlerHooknetCancer
from src.config import HookNetConfig, HookNetDatasetConfig


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
                        default='hooknet_cancer',
                        type=str,
                        help='model name',
                        required=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    ds_config = HookNetDatasetConfig
    config = HookNetConfig
    assert config.OUTPUT_TYPE == 'cancer', "change OUTPUT_TYPE to 'cancer' in HookNetConfig, config.py"
    print(f'Training on levels {ds_config.LEVELS[0]} and {ds_config.LEVELS[1]}.')

    dataset_dir = Path(args.dataset_dir)
    slides_dir = dataset_dir / 'slides'
    contours_dir = dataset_dir / 'contours'

    if not slides_dir.is_dir():
        print('dataset does not exist.')
    else:
        dataset = DatasetHandlerHooknetCancer(slides_dir,
                                              contours_dir,
                                              str(Path(args.subset_file_path)),
                                              ds_config)
        train_data_gen, val_data_gen, n_iter_train, n_iter_val = dataset.create_data_generators()

        checkpoints_dir = Path(os.path.join(args.out_dir, 'checkpoints'))
        logs_dir = Path(os.path.join(args.out_dir, 'logs'))

        hooknet = HookNet(checkpoints_dir=checkpoints_dir,
                          log_dir=logs_dir,
                          name=args.model_name,
                          config=config)

        hooknet.train(train_data_gen, n_iter_train, val_data_gen, n_iter_val, ds_config.EPOCHS)
