"""
Plot the correlation across subjects of a behavioral measure between two
halves of the sessions within a task.
"""
import argparse
import data_analysis_utils as dana
import data_behavioral_measures as bmeas
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import pingouin as pg
import plot_utils as plut
import random
import scipy.stats

FNAME_PREFIX = "corr_across_subs_within_task"
BMEAS_KEYS_CORR_WITHIN_TASKS = [
    "update-freq",
    "coef-z-uncertainty",
    "coef-z-cpp",
    ]

def run_with_args(group_def_file):
    group_def_file = args.group_def_file
    data_files = dana.get_subject_data_files_from_group_def(group_def_file)
    n_subjects = len(data_files)
    output_dir = dana.get_output_dir_for_group_def_file(group_def_file,
        create_if_needed=True)
    task_to_sub_sessions = dana.get_task_to_subject_sessions(data_files)
    n_tasks = len(task_to_sub_sessions)
    tasks = list(task_to_sub_sessions.keys())

    data = dana.load_data(data_files[0])
    
    bmeas_vals_by_key_task = { (k, task): np.empty((n_subjects, 2))
            for k, task in itertools.product(BMEAS_KEYS_CORR_WITHIN_TASKS, tasks)}
    for task, sub_sessions in task_to_sub_sessions.items():
        for i_sub in range(n_subjects):
            sessions = sub_sessions[i_sub]
            for i_half in range(2):
                measures = bmeas.compute_behavioral_measures(sessions[i_half::2])
                for k in BMEAS_KEYS_CORR_WITHIN_TASKS:
                    bmeas_vals_by_key_task[(k, task)][i_sub, i_half] = measures[k]
    

    plut.setup_mpl_style()

    # Compute correlation of the different measures between the two halves within each task
    corrs_data = []
    for task, k in itertools.product(tasks, BMEAS_KEYS_CORR_WITHIN_TASKS):
        use_partial_corr = ("update-freq" not in k)
        vals = bmeas_vals_by_key_task[(k, task)]
        if use_partial_corr:
            df = pd.DataFrame({
                'x1': vals[:, 0],
                'x2': vals[:, 1],
                'covar': np.mean(bmeas_vals_by_key_task['update-freq', task], axis=1)
                })
            partial_corr_stats = pg.partial_corr(df, x='x1', y='x2', covar='covar')
            r = partial_corr_stats['r'].item()
            p = partial_corr_stats['p-val'].item()
        else:
            r, p = scipy.stats.pearsonr(vals[:, 0], vals[:, 1])
        corrs_data += [{'task': task, 'measure': k, 'r': r, 'p': p, 'is_partial_corr': use_partial_corr}]
    corrs_data = pd.DataFrame(corrs_data)
    corrs_data_fpath = dana.get_path(output_dir, FNAME_PREFIX, 'csv')
    plut.save_stats(corrs_data, corrs_data_fpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group_def_file", type=str,
        default=dana.ADA_LEARN_GROUP_DEF_FILE)
    args = parser.parse_args()
    run_with_args(args.group_def_file)
