"""
Analyze subjects' estimate accuracy in terms of the closeness of their estimates
to the normative model estimate.
For plotting: The subject's estimates are aggregated by bin of model estimate
and averaged at the subject level, then statistics are computed at the group level.
For inferential statistics: The Pearson correlation between the subject's
and the model's estimates is calculated at the subject level, then statistical
testing is performed on the correlation coefficients at the group level.
"""

import argparse
import data_analysis_utils as dana
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker
import model_learner as model
import numpy as np
import os.path as op
import pandas as pd
import plot_utils as plut
import scipy.stats

FNAME_PREFIX = "accuracy_subject-estimate_by_model-estimate"
N_BINS = 6
MARKERSIZE = 2

def run_with_args(group_def_file, nbins=N_BINS, ms=MARKERSIZE):
    data_files = dana.get_subject_data_files_from_group_def(group_def_file)
    n_subjects = len(data_files)
    output_dir = dana.get_output_dir_for_group_def_file(group_def_file,
                create_if_needed=True)
    task_to_sub_sessions = dana.get_task_to_subject_sessions(data_files)
    n_tasks = len(task_to_sub_sessions)

    name_refvals = {
        "pearsonr": [0],
        "slope": [0, 1],
        "yval-at-0p5": [],
    }
    
    for task, sub_sessions in task_to_sub_sessions.items():
        # Compute subject's and model estimates
        subs_data = {}
        subs_data["estimate"] = [None for _ in range(n_subjects)]
        subs_data["model_estimate"] = [None for _ in range(n_subjects)]
        for name in name_refvals:
            subs_data[name] = np.empty(n_subjects)
        for i_sub, sessions in enumerate(sub_sessions):
            arrays = dana.get_data_arrays_from_sessions(sessions,
                keys=["estimate", "model_estimate"])
            subs_data["estimate"][i_sub] = arrays["estimate"]
            subs_data["model_estimate"][i_sub] = arrays["model_estimate"]
            linreg_res = scipy.stats.linregress(
                arrays["model_estimate"].flatten(),
                arrays["estimate"].flatten())
            subs_data["pearsonr"][i_sub] = linreg_res.rvalue
            subs_data["slope"][i_sub] = linreg_res.slope
            subs_data["yval-at-0p5"][i_sub] = linreg_res.intercept + linreg_res.slope * 0.5

        # Compute bins of model estimates
        all_model_estimate = np.concatenate(subs_data["model_estimate"]).flatten()
        _, bin_edges = pd.qcut(all_model_estimate, nbins, retbins=True)

        # Compute average per bin over trials for each subject
        for key in ["estimate", "model_estimate"]:
            subs_data[f"{key}_avg_per_bin"] = np.empty((n_subjects, nbins))
        for i_sub in range(n_subjects):
            for i_bin in range(nbins):
                bin_mask = dana.get_bin_mask(subs_data["model_estimate"][i_sub],
                    bin_edges, i_bin)
                for key in ["estimate", "model_estimate"]:
                    subs_data[f"{key}_avg_per_bin"][i_sub, i_bin] = (
                        np.mean(subs_data[key][i_sub][bin_mask]))

        # Compute average over subjects
        group_data = {}
        for key in ["estimate", "model_estimate"]:
            group_data[f"{key}_avg_per_bin"] = np.mean(
                subs_data[f"{key}_avg_per_bin"], axis=0)
            group_data[f"{key}_sem_per_bin"] = scipy.stats.sem(
                subs_data[f"{key}_avg_per_bin"], axis=0)

        plut.setup_mpl_style()

        # Plot group average+-sem estimate by bin of model estimate
        xlabel = "Normative estimate"
        ylabel = "Subjects' estimate"
        ticks = [0.25, 0.5, 0.75]
        figsize = (plut.A4_PAPER_CONTENT_WIDTH / 4,
                   plut.A4_PAPER_CONTENT_WIDTH / 4)
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        for i_bin in range(nbins):
            ax.errorbar(group_data["model_estimate_avg_per_bin"][i_bin],
                group_data["estimate_avg_per_bin"][i_bin],
                group_data["estimate_sem_per_bin"][i_bin],
                fmt='o', ms=ms,
                color=plut.SUBJECT_COLOR,
                ecolor=plut.GRAY_COLOR,
                )
        xlim = ax.get_xlim()
        xdiag = (xlim[0] + (xlim[1] - xlim[0]) * 0.025,
                 xlim[0] + (xlim[1] - xlim[0]) * 0.975)
        ax.plot(xdiag, xdiag, '--', lw=0.5, color=plut.BLACK_COLOR, zorder=-1)
        figname = FNAME_PREFIX
        if nbins != N_BINS:
            figname += "_{nbins}-bins"
        if n_tasks > 1:
            figname += f"_{task}"
        for ext in ["png", "pdf"]:
            figpath = dana.get_path(output_dir, figname, ext)
            plut.save_figure(fig, figpath)

        # Compute group-level statistics on subject vs model estimate
        for name, refvals in name_refvals.items():
            subs_vals = subs_data[name]
            stat_data = {
                f"Mean of {name}": np.mean(subs_vals),
                f"S.e.m. of {name}": scipy.stats.sem(subs_vals),
                f"Standard deviation of {name}": np.std(subs_vals),
            }
            for refval in refvals:
                testres = scipy.stats.ttest_1samp(subs_vals, refval)
                stat_data[f"t-statistic of t-test against {refval}"] = testres.statistic
                stat_data[f"p-value of t-test against {refval}"] = testres.pvalue
            stat_data["dof"] = len(subs_vals) - 1
            stat_data = pd.Series(stat_data)
            stat_name = f"{FNAME_PREFIX}_stats_{name}"
            if n_tasks > 1:
                stat_name += f"_{task}"
            stat_fpath = dana.get_path(output_dir, stat_name, 'csv')
            plut.save_stats(stat_data, stat_fpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group_def_file", type=str,
        default=dana.ADA_LEARN_GROUP_DEF_FILE)
    parser.add_argument("--nbins", type=int, default=N_BINS)
    args = parser.parse_args()
    run_with_args(args.group_def_file,
        nbins=args.nbins)
