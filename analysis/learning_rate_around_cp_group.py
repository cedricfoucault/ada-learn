"""
Analyze learning rates locked around change points at the group level.
"""

import argparse
import data_analysis_utils as dana
import matplotlib.pyplot as plt
import matplotlib.ticker
from mne.stats import permutation_cluster_1samp_test
import numpy as np
import os.path as op
import pandas as pd
import learning_rate_around_cp_subject
import plot_utils as plut
import scipy.stats

FNAME_PREFIX = learning_rate_around_cp_subject.FNAME_PREFIX

def compute_lr_avg_around_cp_per_sub(sub_sessions, window_around_cp,
    estimate_key="estimate",
    estimate_funs=None,
    use_only_trials_with_update=False,
    do_ignore_start_cp=False):
    # For each subject, compute the average learning rate locked around change points,
    # where the learning rate is computed from the estimates retrieved with the
    # given key (by default, this is the subject's estimates) or the given function.
    lr_avg_around_per_sub = []
    for i_sub in range(len(sub_sessions)):
        sessions = sub_sessions[i_sub]
        estimate_fun = estimate_funs[i_sub] if estimate_funs is not None else None
        lrs_around_cp = dana.compute_lrs_around_cp(sessions, window_around_cp,
            estimate_key=estimate_key,
            estimate_fun=estimate_fun,
            use_only_trials_with_update=use_only_trials_with_update,
            do_ignore_start_cp=do_ignore_start_cp)
        lr_avg_around_cp = np.array([np.mean(lrs) for lrs in lrs_around_cp])
        lr_avg_around_per_sub += [lr_avg_around_cp]
    return np.array(lr_avg_around_per_sub)

def test_sig_diff_against_baseline(window_around_cp,
    lr_avg_around_cp_per_sub):
    # Perform statistical test for lr values significantly different from
    # the baseline lr, cluster-corrected across time
    i_dist_0 = np.searchsorted(window_around_cp, 0)
    baseline_indices = np.arange(0, i_dist_0)
    test_indices = np.arange(i_dist_0, window_around_cp.shape[0])
    baseline_dists = window_around_cp[baseline_indices]
    test_dists = window_around_cp[test_indices]
    lr_baseline = np.nanmean(lr_avg_around_cp_per_sub[:, baseline_indices])
    lr_avg_test = lr_avg_around_cp_per_sub[:, test_indices]
    t_obs, clusters_indices, cluster_p_values, _ = permutation_cluster_1samp_test(
        (lr_avg_test - lr_baseline),
        n_permutations=1e4)
    clusters_indices = [c[0] for c in clusters_indices]
    clusters_sig_dists = [test_dists[c_i] for c_i in clusters_indices]
    return clusters_sig_dists

def run_with_args(group_def_file,
    use_only_trials_with_update=False,
    skip_stat=False,
    skip_model=False,
    do_ignore_start_cp=False):
    """
    Parameters:
    - use_only_trials_with_update:
        If True, all data points where the subject did not perform an update
        (yielding a learning rate equal to 0) will be discarded from the analysis.
    - skip_stat:
        By default, we perform a cluster-corrected statistical test against
        a baseline learning rate (computed right before the change point),
        and plot the results of statistical testing as stars at the top of the graph.
        Set this argument to True to skip this.
    - skip_model:
        By default, we compute, in addition to the subjects' results, the results
        predicted by the normative model, i.e. the results we would get if all
        subjects were behaving exactly as the normative model does.
        Set this argument to True to skip computing those results.
    """
    is_wip = (do_ignore_start_cp)
    data_files = dana.get_subject_data_files_from_group_def(group_def_file)
    output_dir = dana.get_output_dir_for_group_def_file(group_def_file,
        create_if_needed=True, is_wip=is_wip)
    task_to_sub_sessions = dana.get_task_to_subject_sessions(data_files)
    n_tasks = len(task_to_sub_sessions)
    
    for task, sub_sessions in task_to_sub_sessions.items():
        # Compute average learning rate around cp for each subject
        window_around_cp = dana.get_window_around_cp_for_sessions(sub_sessions[0])
        sub_lr_avg_around_cp_per_sub = compute_lr_avg_around_cp_per_sub(
            sub_sessions, window_around_cp,
            use_only_trials_with_update=use_only_trials_with_update)
        if not skip_model:
            model_lr_avg_around_cp_per_sub = compute_lr_avg_around_cp_per_sub(
                sub_sessions, window_around_cp,
                estimate_key="model_estimate",
                use_only_trials_with_update=use_only_trials_with_update)

        # Compute mean and s.e.m. across subjects
        sub_lr_avg_around_cp_groupmean = np.nanmean(sub_lr_avg_around_cp_per_sub, axis=0)
        sub_lr_avg_around_cp_groupsem = scipy.stats.sem(sub_lr_avg_around_cp_per_sub, axis=0,
            nan_policy="omit")
        if not skip_model:
            model_lr_avg_around_cp_groupmean = np.nanmean(model_lr_avg_around_cp_per_sub, axis=0)

        # Perform cluster-corrected statistical test
        if not skip_stat:
            clusters_sig_dists = test_sig_diff_against_baseline(
                window_around_cp, sub_lr_avg_around_cp_per_sub)

        # Plot and save outputs
        plut.setup_mpl_style()

        fig = learning_rate_around_cp_subject.make_figure(window_around_cp,
            sub_lr_avg_around_cp_groupmean, sub_lr_avg_around_cp_groupsem,
            task=task,
            clusters_sig_dists=(clusters_sig_dists if not skip_stat else None),
            use_only_trials_with_update=use_only_trials_with_update,
            do_ignore_start_cp=do_ignore_start_cp)

        def get_fname(prefix):
            fname = prefix
            if use_only_trials_with_update:
                fname += "_use-only-trials-with-update"
            if skip_stat:
                fname += "_no-stat"
            if do_ignore_start_cp:
                fname += "_do-ignore-start-cp"
            if n_tasks > 1:
                fname += f"_{task}"
            return fname

        figname = get_fname(FNAME_PREFIX)
        for ext in ['png', 'pdf']:
            figpath = dana.get_path(output_dir, figname, ext)
            plut.save_figure(fig, figpath, do_fit_axis_labels=True)

        if not skip_model:
            fig = learning_rate_around_cp_subject.make_figure(window_around_cp,
                model_lr_avg_around_cp_groupmean, None,
                task=task, color=plut.BLACK_COLOR)
            figname_model = get_fname(FNAME_PREFIX + "_model")
            for ext in ['pdf']:
                figpath = dana.get_path(output_dir, figname_model, ext)
                plut.save_figure(fig, figpath, do_fit_axis_labels=True)

        if not skip_model:
            # Compute the correlation between the subjects' curve and the normative model's curve
            # of the learning rate around change points
            group_model_correlation_r, group_model_correlation_p = scipy.stats.pearsonr(
                sub_lr_avg_around_cp_groupmean,
                model_lr_avg_around_cp_groupmean)
            group_model_correlation_data = pd.Series({
                "Pearson_correlation_r": group_model_correlation_r,
                "Pearson_correlation_p": group_model_correlation_p,
                })
            group_model_correlation_fname = get_fname(FNAME_PREFIX + "_corr-with-model")
            group_model_correlation_fpath = dana.get_path(output_dir,
                group_model_correlation_fname, 'csv')
            group_model_correlation_data.to_csv(group_model_correlation_fpath)
            print(f"Corr. between subjects' curve and normative model's curve "
                  f"saved at {group_model_correlation_fpath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group_def_file", type=str,
        default=dana.ADA_LEARN_GROUP_DEF_FILE)
    parser.add_argument("--use_only_trials_with_update", action="store_true", default=False)
    parser.add_argument("--skip_stat", action="store_true", default=False)
    parser.add_argument("--skip_model", action="store_true", default=False)
    parser.add_argument("--do_ignore_start_cp", action="store_true", default=False)
    args = parser.parse_args()
    run_with_args(args.group_def_file,
        skip_stat=args.skip_stat,
        skip_model=args.skip_model,
        use_only_trials_with_update=args.use_only_trials_with_update,
        do_ignore_start_cp=args.do_ignore_start_cp)
