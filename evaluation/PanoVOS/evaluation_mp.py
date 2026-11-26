from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from PanoVOS.evaluation import (
    PanoVOSEvaluation as _BaseEvaluation,
    accumulate_sequence_metrics,
    init_metric_containers,
)
from PanoVOS.results import Results

_WORKER_STATE = {}


def _init_eval_worker(root, task, gt_set, codalab, res_path, metric):
    _WORKER_STATE['eval'] = _BaseEvaluation(PanoVOS_root=root, task=task, gt_set=gt_set, codalab=codalab)
    _WORKER_STATE['results'] = Results(root_dir=res_path)
    _WORKER_STATE['metric'] = metric


def _eval_sequence_worker(seq_key):
    ev = _WORKER_STATE['eval']
    seq = ev.dataset.get_sequence(seq_key)
    _seq_name = list(seq.keys())[0]
    objs = list(seq[_seq_name])
    j_res, f_res = ev._evaluate_semisupervised(seq, _WORKER_STATE['results'], None, _WORKER_STATE['metric'])
    return _seq_name, objs, j_res, f_res


class PanoVOSEvaluation(_BaseEvaluation):
    def __init__(self, PanoVOS_root, task, gt_set, mp_procs=8, codalab=False):
        super().__init__(PanoVOS_root=PanoVOS_root, task=task, gt_set=gt_set, codalab=codalab)
        self.mp_procs = mp_procs

    def evaluate(self, res_path, metric=('J', 'F'), debug=False):
        metric = metric if isinstance(metric, (tuple, list)) else [metric]
        if 'T' in metric:
            raise ValueError('Temporal metric not supported!')
        if 'J' not in metric and 'F' not in metric:
            raise ValueError('Metric possible values are J for IoU or F for Boundary')

        metrics_res, metrics_res_seen, metrics_res_unseen = init_metric_containers(metric)
        sequences = list(self.dataset.get_sequences())

        with ProcessPoolExecutor(
            max_workers=self.mp_procs,
            initializer=_init_eval_worker,
            initargs=(self.PanoVOS_root, self.task, self.dataset.subset, False, res_path, metric),
        ) as pool:
            for _seq_name, objs, j_res, f_res in tqdm(
                pool.map(_eval_sequence_worker, sequences),
                total=len(sequences),
                desc='Eval PanoVOS',
            ):
                accumulate_sequence_metrics(
                    _seq_name,
                    objs,
                    j_res,
                    f_res,
                    _seq_name in self.unseen_videos,
                    metric,
                    metrics_res,
                    metrics_res_seen,
                    metrics_res_unseen,
                )

        return metrics_res, metrics_res_seen, metrics_res_unseen
