from __future__ import annotations

import os
from pathlib import Path

from tomato_model_utils import compare_dataset_sources, dataset_ready_for_training


REPO_ROOT = Path(__file__).resolve().parent
LOCAL_DATASET = Path(os.environ.get('LOCAL_DATASET', REPO_ROOT.parent / 'Tomato_Leaf_Dataset'))
KAGGLE_DATASET = Path(os.environ.get('KAGGLE_DATASET', LOCAL_DATASET))


def main() -> None:
    """Print a concise dataset comparison for the two tomato leaf collections."""
    if not LOCAL_DATASET.exists():
        raise FileNotFoundError(f'Local dataset folder not found: {LOCAL_DATASET}')

    if not dataset_ready_for_training(str(LOCAL_DATASET)):
        raise FileNotFoundError(f'No image data was found in {LOCAL_DATASET}')

    comparison = compare_dataset_sources(str(LOCAL_DATASET), str(KAGGLE_DATASET))
    print('Tomato leaf disease dataset comparison')
    print('=' * 40)
    print('Local dataset summary:')
    print(comparison['local_dataset'])
    print('\nKaggle dataset summary:')
    print(comparison['kaggle_dataset'])
    print('\nRecommended model strategy:')
    print(comparison['recommendation'])

    print('\nArchitecture choice: MobileNetV3-Small/Large')
    print('Reason: lightweight, fast inference, and better deployment for web-level disease detection than a traditional CNN baseline.')


if __name__ == '__main__':
    main()
