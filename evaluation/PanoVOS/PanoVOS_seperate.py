import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image


class PanoVOS(object):
    SUBSET_OPTIONS = ['val', 'test']
    TASKS = ['semi-supervised']
    DATASET_WEB = 'https://github.com/shilinyan99/PanoVOS'
    VOID_LABEL = 255

    def __init__(self, root, task='semi-supervised', subset='val', codalab=False):
        """Class to read the PanoVOS ground-truth annotations and metadata.

        :param root: Path to either the PanoVOS root folder (containing val/ and test/)
                     or directly to the subset folder (containing Annotations/ and {subset}_meta.json).
        :param task: Task to load the annotations ('semi-supervised').
        :param subset: Subset to evaluate ('val' or 'test').
        """
        if subset not in self.SUBSET_OPTIONS:
            raise ValueError(f'Subset should be in {self.SUBSET_OPTIONS}')
        if task not in self.TASKS:
            raise ValueError(f'The only tasks that are supported are {self.TASKS}')

        self.task = task
        self.subset = subset
        self.root = root

        # Support both {PanoVOS_ROOT} (with val/ or test/ child directory) and direct subset directory
        candidate_subset_root = os.path.join(root, subset)
        if os.path.isfile(os.path.join(candidate_subset_root, f'{subset}_meta.json')):
            self.subset_root = candidate_subset_root
        else:
            self.subset_root = root

        annotations_folder = 'Annotations' if task == 'semi-supervised' else 'Annotations_unsupervised'
        self.mask_path = os.path.join(self.subset_root, annotations_folder)
        json_path = os.path.join(self.subset_root, f'{self.subset}_meta.json')

        if not os.path.isfile(json_path):
            raise FileNotFoundError(
                f'Metadata file not found: {json_path}\n'
                f'Please check --PanoVOS_path or omit --PanoVOS_path to use the bundled annotations/{subset}_annotations.zip.'
            )
        if not os.path.isdir(self.mask_path):
            raise FileNotFoundError(
                f'Annotations directory not found: {self.mask_path}\n'
                f'Please extract annotations/{subset}_annotations.zip or omit --PanoVOS_path.'
            )

        with open(json_path, 'r') as f:
            self.json_data = json.load(f)['videos']

        self.sequences_names = sorted(self.json_data.keys())
        self.sequences = defaultdict(dict)
        for seq in self.sequences_names:
            self.sequences[seq] = self.json_data[seq]["objects"]

    def get_sequences(self):
        for seq in self.sequences:
            yield seq

    def get_sequence(self, sequence):
        return {sequence: self.sequences[sequence]}

    def get_raw_mask(self, sequence, frame):
        mask_file = os.path.join(self.mask_path, sequence, frame + '.png')
        if not os.path.isfile(mask_file):
            raise FileNotFoundError(
                f'Ground-truth mask not found: {mask_file}\n'
                f'Note: valid.zip / test.zip on Google Drive only include 1st-frame annotations. '
                f'For full-frame ground truth, extract annotations/{self.subset}_annotations.zip '
                f'into your {self.subset} directory (or omit --PanoVOS_path).'
            )
        return np.array(Image.open(mask_file))

    def get_mask(self, sequence, frame, target_obj=None):
        masks = np.expand_dims(self.get_raw_mask(sequence, frame), axis=0)
        masks_void = np.zeros_like(masks)

        if target_obj is not None:
            tmp_masks = np.zeros_like(masks)
            tmp_masks[masks == int(target_obj)] = 1
            masks = tmp_masks

        for i in range(masks.shape[0]):
            masks_void[i, ...] = masks[i, ...] == self.VOID_LABEL
            masks[i, masks[i, ...] == self.VOID_LABEL] = 0

        return masks, masks_void
