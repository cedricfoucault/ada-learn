"""
Plot the distribution of intervals (in terms of number of outcomes, or trials)
between two estimate updates that the subjects performed.

Note: The interval between two updates corresponds to what is called "step width"
in Gallistel et al. (2014).
"""

import argparse
import data_analysis_utils as dana
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import os.path as op
import pandas as pd
import plot_utils as plut

FNAME_PREFIX = "update_interval_hist"

FIGSIZE = (plut.A4_PAPER_CONTENT_WIDTH * 1/2, plut.DEFAULT_HEIGHT)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group_def_file", type=str,
        default=dana.ADA_LEARN_GROUP_DEF_FILE)
    args = parser.parse_args()
    group_def_file = args.group_def_file
    data_files = dana.get_subject_data_files_from_group_def(group_def_file)
    output_dir = dana.get_output_dir_for_group_def_file(group_def_file,
        create_if_needed=True)
    n_subjects = len(data_files)
    task_to_sub_sessions = dana.get_task_to_subject_sessions(data_files)
    n_tasks = len(task_to_sub_sessions)

    plut.setup_mpl_style()
    
    for task, sub_sessions in task_to_sub_sessions.items():
        # Compute distribution of intervals across all sessions and subjects
        dist_between_updates_all = []
        for i_sub, sessions in enumerate(sub_sessions):
            measures = dana.estimate_update_measures_from_sessions(sessions)
            was_estimate_updated = measures["was_estimate_updated"]
            n_sessions, n_trials = was_estimate_updated.shape
            for i_sess in range(n_sessions):
                update_indices = np.arange(n_trials)[was_estimate_updated[i_sess]]
                dist_between_updates = np.diff(np.insert(update_indices, 0, -1))
                dist_between_updates_all += list(dist_between_updates)

        # Compute and plot histogram
        N_BINS = 18 # This yields about the same bins as Gallistel et al., 2014, Fig. 11
                    # when truncating the histogram at 75 outcomes
                    # (which is the length of a session in our study).
        logbin_centers = np.logspace(np.log10(1), np.log10(n_trials), N_BINS).round().astype(int)
        logbin_centers = np.unique(logbin_centers) # remove duplicates (this can happen due to integer rounding)
        logbin_edges = np.insert(logbin_centers, 0, 0) + 0.5
        xticks = [1, 3, 10, 32, 75]
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.gca()
        ax.set_xlabel("Observations between two updates")
        ax.set_xscale("log", subs=[])
        ax.minorticks_off()
        ax.set_xticks(xticks, [f"{c:d}" for c in xticks])
        ax.xaxis.set_major_formatter("{x:d}")
        ax.set_ylabel("Proportion")
        weights = (np.ones(len(dist_between_updates_all))
            / len(dist_between_updates_all)) # to use proportions instead of counts
        props, _, _ = ax.hist(dist_between_updates_all,
            weights=weights,
            bins=logbin_edges,
            color=plut.SUBJECT_COLOR)
        figname = FNAME_PREFIX
        if n_tasks > 1:
            figname += f"_{task}"
        for ext in ['png', 'pdf']:
            figpath = dana.get_path(output_dir, figname, ext)
            plut.save_figure(fig, figpath)

        # Save summary statistics of the distribution
        stat_name = FNAME_PREFIX
        if n_tasks > 1:
            stat_name += f"_{task}"
        stat_fpath = dana.get_path(output_dir, stat_name, 'csv')
        stat_data = pd.Series({
                "mean_interval": np.mean(dist_between_updates_all),
                "pct_interval-equal-to-one": props[0] * 100,
            })
        stat_data.to_csv(stat_fpath)
        print(f"Saved stats at {stat_fpath}")

