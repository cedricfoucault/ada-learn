"""
The objective of this analysis is to assess the reliability of the results
when using a reduced number of sessions, in order to determine a minimum
number of sessions to perform in the study.

We assess reliability of a scalar measurement in three ways:
1) The correlation across subjects of the measurements obtained with the reduced
number of sessions and the maximum available number of sessions.
2) The error (absolute difference) in the group mean of the measurement
obtained with the reduced number of sessions vs. with the maximum number of sessions.
3) The mean across subjects of the error (absolute difference) in the subject mean
between that obtained with the reduced and that obtained with the maximum number of sessions.

For a vector measurement, we do 2) and 3) but instead of taking the absolute
difference for the error, we take the root mean square error (euclidean distance)
or the pearson correlation.
"""

import argparse
import data_analysis_utils as dana
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import os.path as op
import learning_rate_relationship_cpp_uncertainty_subject
import zz_learning_rate_curve_measures_subject
import performance_over_sessions_subject
import plot_utils as plut
import scipy.stats

FIGSIZE = (plut.A4_PAPER_CONTENT_WIDTH / 2, plut.DEFAULT_HEIGHT * 1.5)

def run_with_args(group_def_file,
    n_sessions_min=2):
    data_files = dana.get_subject_data_files_from_group_def(group_def_file)
    output_dir = dana.get_output_dir_for_group_def_file(group_def_file,
        create_if_needed=True, is_wip=True)
    n_subjects = len(data_files)
    n_sessions_max = None
    meas_keys = ["corr_uncert", "lr_diff_close-far", "corr_nodelay", "maxcorr",
        "performance"]
    n_meas = len(meas_keys)
    # for each measurement indexed by meas_key, subs_data contains a 2d array
    # of shape (n_subjects, n_sessions_max - n_sessions_min + 1)
    subs_data = {k: [] for k in meas_keys}
    for i_sub, data_file in enumerate(data_files):
        data = dana.load_data(data_file)
        all_sessions = data["taskSessionDataArray"]
        if n_sessions_max is None:
            n_sessions_max = len(all_sessions)
            n_sessions_range = np.arange(n_sessions_min, n_sessions_max + 1)
        sub_data = {k: [] for k in meas_keys}
        for n_sessions in n_sessions_range:
            reduced_sessions = all_sessions[:n_sessions]
            res = zz_learning_rate_curve_measures_subject.run_with_sessions(
                reduced_sessions, do_plot=False)
            for k in ["lr_diff_close-far", "corr_nodelay", "maxcorr"]:
                sub_data[k] += [res[k]]
            res = learning_rate_relationship_cpp_uncertainty_subject.run_with_sessions(
                reduced_sessions, do_plot=False)
            sub_data["corr_uncert"] += [res["pearsonr_uncertainty"]]
            res = performance_over_sessions_subject.run_with_sessions(
                reduced_sessions, do_plot=False)
            sub_data["performance"] += [np.mean(res["performances"])]
        for k in meas_keys:
            subs_data[k] += [sub_data[k]]
    for k in meas_keys:
        subs_data[k] = np.array(subs_data[k])

    # 1) Correlation across subjects of the measurements obtained with the reduced
    # number of sessions and the maximum available number of sessions.
    corrs_reduced_max = {k: np.empty(len(n_sessions_range)) for k in meas_keys}
    # 2) Error (absolute difference) in the group mean, i.e., the error
    # between the mean across subjects of the measurement obtained with the reduced
    # number of sessions and the mean across subjects of the measurement obtained with
    # the maximum number of sessions.
    errors_group_mean = {(k, is_relative): np.empty(len(n_sessions_range))
        for k in meas_keys for is_relative in [False, True]}
    # 3) Mean across subjects of the error (absolute difference) in the subject mean,
    # i.e., the mean across subjects of the error in the measurement
    # between the reduced and the maximum number of sessions.
    errors_sub_mean = {(k, is_relative): np.empty(len(n_sessions_range))
        for k in meas_keys for is_relative in [False, True]}
    for k in meas_keys:
        for i, n_sessions in enumerate(n_sessions_range):
            subvals_reduced = subs_data[k][:, i]
            subvals_max = subs_data[k][:, -1]
            corr = scipy.stats.pearsonr(
                subvals_reduced,
                subvals_max)[0]
            corrs_reduced_max[k][i] = corr
            err_group = np.abs(
                np.mean(subvals_reduced)
                - np.mean(subvals_max))
            errors_group_mean[(k, False)][i] = err_group
            errors_group_mean[(k, True)][i] = (err_group
                / np.abs(np.mean(subvals_max)))
            err_subs = np.abs(subvals_reduced - subvals_max)
            errors_sub_mean[(k, False)][i] = np.mean(err_subs)
            errors_sub_mean[(k, True)][i] = np.mean(err_subs
                / np.abs(subvals_max))
            
    plut.setup_mpl_style()

    xlabel = "Reduced num. of sessions"
    ylabel = ""
    xmargin, ymargin = plt.margins()
    ylim_corr = (0 - ymargin, 1 + ymargin)
    ylim_rel_error = (0 - ymargin, 1 + ymargin)
    ylim_abs_error = (0 - ymargin, None)
    for k in meas_keys:
        for yvals, figname, title, ylim in [
            (corrs_reduced_max[k],
            f"reduced_sessions_{k}_correlation",
            f"Correlation in {k} obtained with\nthe reduced and max num. of sessions",
            ylim_corr),
            (errors_group_mean[(k, False)],
            f"reduced_sessions_{k}_group_mean_error-abs",
            f"Error in group mean of {k} between\nthe reduced and max num. of sessions",
            ylim_abs_error),
            (errors_group_mean[(k, True)],
            f"reduced_sessions_{k}_group_mean_error-rel",
            f"Relative error in group mean of {k} between\nthe reduced and max num. of sessions",
            ylim_rel_error),
            (errors_sub_mean[(k, False)],
            f"reduced_sessions_{k}_subject_mean_error-abs",
            f"Error in subject's mean of {k} between\nthe reduced and max num. of sessions",
            ylim_abs_error),
            (errors_sub_mean[(k, True)],
            f"reduced_sessions_{k}_subject_mean_error-rel",
            f"Relative error in subject's mean of {k} between\nthe reduced and max num. of sessions",
            ylim_rel_error),]:
    
            fig = plt.figure(figsize=FIGSIZE)
            ax = fig.gca()
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_ylim(ylim)
            ax.set_title(title)
            ax.plot(n_sessions_range, yvals, '-', color=plut.COLORS_NEUTRAL_GRAY[0])
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            figname += f"_nsess-min-{n_sessions_min}"
            figpath = dana.get_path(output_dir, figname)
            plut.save_figure(fig, figpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group_def_file")
    parser.add_argument("--n_sessions_min", type=int, default=2)
    args = parser.parse_args()
    run_with_args(args.group_def_file,
        n_sessions_min=args.n_sessions_min)

