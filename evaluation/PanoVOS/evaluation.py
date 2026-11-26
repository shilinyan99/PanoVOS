import os
import warnings
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from PanoVOS import utils
from PanoVOS.PanoVOS_seperate import PanoVOS
from PanoVOS.metrics import db_eval_boundary, db_eval_iou
from PanoVOS.results import Results

warnings.filterwarnings("ignore", category=RuntimeWarning)


def resolve_unseen_list(subset_root, gt_set):
    eval_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(subset_root, f'{gt_set}_unseen.txt'),
        os.path.join(eval_dir, f'{gt_set}_unseen.txt'),
        f'{gt_set}_unseen.txt',
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path, mode='r') as f:
                return {line.strip() for line in f if line.strip()}
    raise FileNotFoundError(f'Could not find {gt_set}_unseen.txt in {candidates}')


def init_metric_containers(metric):
    metrics_res = {}
    metrics_res_seen = {}
    metrics_res_unseen = {}
    for m in ('J', 'F'):
        if m in metric:
            metrics_res[m] = {"M": [], "R": [], "D": [], "M_per_object": {}}
            metrics_res_seen[m] = {"M": [], "R": [], "D": [], "M_per_object": {}}
            metrics_res_unseen[m] = {"M": [], "R": [], "D": [], "M_per_object": {}}
    return metrics_res, metrics_res_seen, metrics_res_unseen


def accumulate_sequence_metrics(
    seq_name,
    objs,
    j_metrics_res,
    f_metrics_res,
    is_unseen,
    metric,
    metrics_res,
    metrics_res_seen,
    metrics_res_unseen,
):
    target = metrics_res_unseen if is_unseen else metrics_res_seen
    for ii, _obj in enumerate(objs):
        obj_seq_name = f'{seq_name}_{ii + 1}'
        if 'J' in metric:
            JM, JR, JD = utils.db_statistics(j_metrics_res[str(_obj)])
            metrics_res['J']["M"].append(JM)
            metrics_res['J']["R"].append(JR)
            metrics_res['J']["D"].append(JD)
            metrics_res['J']["M_per_object"][obj_seq_name] = JM
            target['J']["M"].append(JM)
            target['J']["R"].append(JR)
            target['J']["D"].append(JD)
            target['J']["M_per_object"][obj_seq_name] = JM
        if 'F' in metric:
            FM, FR, FD = utils.db_statistics(f_metrics_res[str(_obj)])
            metrics_res['F']["M"].append(FM)
            metrics_res['F']["R"].append(FR)
            metrics_res['F']["D"].append(FD)
            metrics_res['F']["M_per_object"][obj_seq_name] = FM
            target['F']["M"].append(FM)
            target['F']["R"].append(FR)
            target['F']["D"].append(FD)
            target['F']["M_per_object"][obj_seq_name] = FM


class PanoVOSEvaluation(object):
    def __init__(self, PanoVOS_root, task, gt_set, codalab=False):
        self.PanoVOS_root = PanoVOS_root
        self.task = task
        self.dataset = PanoVOS(root=PanoVOS_root, task=task, subset=gt_set, codalab=codalab)
        self.unseen_videos = resolve_unseen_list(self.dataset.subset_root, gt_set)

    def _evaluate_semisupervised(self, seq, results, all_void_masks, metric):
        seq_name = list(seq.keys())[0]
        seq = seq[seq_name]
        objs = list(seq.keys())

        j_metrics_res = {}
        f_metrics_res = {}
        for _obj in objs:
            _frame_num = int(seq[_obj]['frame_range']['frame_nums'])
            j_metrics_res[str(_obj)] = np.zeros((1, _frame_num))
            f_metrics_res[str(_obj)] = np.zeros((1, _frame_num))

        frame_to_objs = defaultdict(list)
        for _obj in objs:
            start_frame = int(seq[_obj]['frame_range']['start'])
            end_frame = int(seq[_obj]['frame_range']['end'])
            obj_key = str(_obj)
            obj_int = int(_obj)
            for oidx, ii in enumerate(range(start_frame, end_frame, 5)):
                frame_to_objs[f"{ii:05d}"].append((obj_key, obj_int, oidx))

        for frame_str, active_objs in frame_to_objs.items():
            gt_raw = self.dataset.get_raw_mask(seq_name, frame_str)
            res_raw = results._read_mask(seq_name, frame_str)
            for obj_key, obj_int, oidx in active_objs:
                gt_mask = (gt_raw == obj_int)[None, ...]
                res_mask = (res_raw == obj_int)[None, ...]
                if 'J' in metric:
                    j_metrics_res[obj_key][0, oidx] = np.asarray(db_eval_iou(gt_mask, res_mask, all_void_masks)).item()
                if 'F' in metric:
                    f_metrics_res[obj_key][0, oidx] = np.asarray(db_eval_boundary(gt_mask, res_mask, all_void_masks)).item()
        return j_metrics_res, f_metrics_res

    def evaluate(self, res_path, metric=('J', 'F'), debug=False):
        metric = metric if isinstance(metric, (tuple, list)) else [metric]
        if 'T' in metric:
            raise ValueError('Temporal metric not supported!')
        if 'J' not in metric and 'F' not in metric:
            raise ValueError('Metric possible values are J for IoU or F for Boundary')

        metrics_res, metrics_res_seen, metrics_res_unseen = init_metric_containers(metric)
        results = Results(root_dir=res_path)

        for seq_key in tqdm(list(self.dataset.get_sequences()), desc='Eval PanoVOS'):
            seq = self.dataset.get_sequence(seq_key)
            _seq_name = list(seq.keys())[0]
            objs = list(seq[_seq_name])
            is_unseen = _seq_name in self.unseen_videos
            j_metrics_res, f_metrics_res = self._evaluate_semisupervised(seq, results, None, metric)
            accumulate_sequence_metrics(
                _seq_name,
                objs,
                j_metrics_res,
                f_metrics_res,
                is_unseen,
                metric,
                metrics_res,
                metrics_res_seen,
                metrics_res_unseen,
            )

        return metrics_res, metrics_res_seen, metrics_res_unseen
