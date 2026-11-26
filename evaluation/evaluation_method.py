#!/usr/bin/env python3
import argparse
import os
import sys
import tempfile
import zipfile
from time import time

import numpy as np
import pandas as pd

# Allow running from repo root (`python evaluation/evaluation_method.py ...`) or inside `evaluation/`
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from PanoVOS.evaluation import PanoVOSEvaluation as PanoVOSEvaluation_SP
from PanoVOS.evaluation_mp import PanoVOSEvaluation as PanoVOSEvaluation_MP


def ensure_default_gt_path(subset):
    """Resolve bundled annotations/{subset} directory, auto-extracting the zip on first run."""
    annotations_dir = os.path.join(REPO_ROOT, 'annotations')
    target_dir = os.path.join(annotations_dir, subset)
    meta_file = os.path.join(target_dir, f'{subset}_meta.json')
    if os.path.isfile(meta_file) and os.path.isdir(os.path.join(target_dir, 'Annotations')):
        return target_dir

    zip_path = os.path.join(annotations_dir, f'{subset}_annotations.zip')
    if os.path.isfile(zip_path):
        print(f'Extracting bundled {subset} ground-truth annotations from {zip_path} ...')
        os.makedirs(target_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target_dir)
        return target_dir

    raise FileNotFoundError(
        f'Bundled ground-truth archive not found at {zip_path}. '
        f'Please specify --PanoVOS_path pointing to your PanoVOS dataset directory.'
    )


def resolve_predictions_dir(raw_path, expected_sequences):
    """Resolve prediction sequence root from either `<dir>/<seq>/` or `<dir>/Annotations/<seq>/`."""
    if not os.path.exists(raw_path):
        sys.exit(f'Error: results_path does not exist: {raw_path}')

    sample_seq = expected_sequences[0]
    if os.path.isdir(os.path.join(raw_path, sample_seq)):
        return raw_path
    nested = os.path.join(raw_path, 'Annotations')
    if os.path.isdir(nested):
        return nested
    return raw_path


def validate_sequences_present(pred_dir, expected_sequences):
    existing = set(os.listdir(pred_dir)) if os.path.isdir(pred_dir) else set()
    missing = [seq for seq in expected_sequences if seq not in existing]
    if missing:
        sys.stderr.write(
            f'Error: {len(missing)}/{len(expected_sequences)} sequence folder(s) missing in {pred_dir}:\n'
            f'  {", ".join(missing[:10])}{" ..." if len(missing) > 10 else ""}\n'
            'Expected structure: <results_path>/<video_id>/<frame_id>.png '
            '(or <results_path>/Annotations/<video_id>/<frame_id>.png).\n'
        )
        sys.exit(1)


def main():
    time_start = time()
    parser = argparse.ArgumentParser(description='PanoVOS Official Local Evaluation')
    parser.add_argument(
        '--PanoVOS_path',
        type=str,
        default=None,
        help='Path to PanoVOS root or subset folder. If omitted, uses bundled annotations/{set}_annotations.zip.',
    )
    parser.add_argument('--set', type=str, default='val', choices=['val', 'test'], help='Subset to evaluate (val or test)')
    parser.add_argument('--mp_nums', type=int, default=8, help='Number of worker processes (default: 8; 1 for single-process)')
    parser.add_argument('--task', type=str, default='semi-supervised', help='Task to evaluate (default: semi-supervised)')
    parser.add_argument(
        '--results_path',
        type=str,
        required=True,
        help='Path to prediction directory (<results_path>/<video_id>/<frame>.png) or a submission .zip file',
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Directory to save CSV results (defaults to --results_path, or current directory if --results_path is a .zip)',
    )
    parser.add_argument('--recompute', action='store_true', help='Recompute metrics even if cached CSV files exist')
    args = parser.parse_args()

    gt_path = args.PanoVOS_path if args.PanoVOS_path is not None else ensure_default_gt_path(args.set)

    is_zip = os.path.isfile(args.results_path) and zipfile.is_zipfile(args.results_path)
    if args.output_path is not None:
        output_dir = args.output_path
    elif is_zip:
        output_dir = os.path.dirname(os.path.abspath(args.results_path)) or '.'
    else:
        output_dir = args.results_path
    os.makedirs(output_dir, exist_ok=True)

    csv_name_global_path = os.path.join(output_dir, f'global_results-{args.set}.csv')
    csv_name_per_sequence_path = os.path.join(output_dir, f'per-sequence_results-{args.set}.csv')

    if not args.recompute and os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path):
        print(f'Using precomputed results in {output_dir} (pass --recompute to force re-evaluation)...')
        table_g = pd.read_csv(csv_name_global_path)
        table_seq = pd.read_csv(csv_name_per_sequence_path)
    else:
        if args.mp_nums <= 1:
            print('Evaluating with single process...')
            dataset_eval = PanoVOSEvaluation_SP(PanoVOS_root=gt_path, task=args.task, gt_set=args.set)
        else:
            print(f'Evaluating with {args.mp_nums} worker processes...')
            dataset_eval = PanoVOSEvaluation_MP(
                PanoVOS_root=gt_path,
                task=args.task,
                gt_set=args.set,
                mp_procs=args.mp_nums,
            )

        expected_sequences = list(dataset_eval.dataset.get_sequences())
        temp_ctx = tempfile.TemporaryDirectory() if is_zip else None
        try:
            if is_zip:
                print(f'Extracting submission archive {args.results_path} ...')
                with zipfile.ZipFile(args.results_path, 'r') as zf:
                    zf.extractall(temp_ctx.name)
                raw_results_dir = temp_ctx.name
            else:
                raw_results_dir = args.results_path

            pred_dir = resolve_predictions_dir(raw_results_dir, expected_sequences)
            validate_sequences_present(pred_dir, expected_sequences)

            metrics_res, metrics_res_seen, metrics_res_unseen = dataset_eval.evaluate(pred_dir)
        finally:
            if temp_ctx is not None:
                temp_ctx.cleanup()

        J, F = metrics_res['J'], metrics_res['F']
        J_seen, F_seen = metrics_res_seen['J'], metrics_res_seen['F']
        J_unseen, F_unseen = metrics_res_unseen['J'], metrics_res_unseen['F']

        g_measures = ['J&F-Mean', 'J-Mean', 'J-seen-Mean', 'J-unseen-Mean', 'F-Mean', 'F-seen-Mean', 'F-unseen-Mean']
        final_mean = (
            (np.mean(J_seen["M"]) + np.mean(F_seen["M"]))
            + (np.mean(J_unseen["M"]) + np.mean(F_unseen["M"]))
        ) / 4.0

        g_res = np.array([
            final_mean,
            (np.mean(J_seen["M"]) + np.mean(J_unseen["M"])) / 2.0,
            np.mean(J_seen["M"]),
            np.mean(J_unseen["M"]),
            (np.mean(F_seen["M"]) + np.mean(F_unseen["M"])) / 2.0,
            np.mean(F_seen["M"]),
            np.mean(F_unseen["M"]),
        ]).reshape(1, -1)

        table_g = pd.DataFrame(data=g_res, columns=g_measures)
        table_g.to_csv(csv_name_global_path, index=False, float_format="%.3f")
        print(f'Global results saved in {csv_name_global_path}')

        seq_names = list(J['M_per_object'].keys())
        seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
        J_per_object = [J['M_per_object'][x] for x in seq_names]
        F_per_object = [F['M_per_object'][x] for x in seq_names]
        table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
        table_seq.to_csv(csv_name_per_sequence_path, index=False, float_format="%.3f")
        print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

    sys.stdout.write(f"\n--------------------------- Global results for {args.set} ---------------------------\n")
    print(table_g.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    sys.stdout.write(f"\n---------- Per sequence results for {args.set} ----------\n")
    print(table_seq.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    sys.stdout.write(f"\nTotal time: {time() - time_start:.2f}s\n")


if __name__ == '__main__':
    main()
