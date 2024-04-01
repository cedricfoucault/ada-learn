"""
Analyze relationship between the subject's learning rate
and model variable (cpp, uncertainty) at the group level.
"""

import argparse
import data_analysis_utils as dana
import matplotlib.pyplot as plt
import matplotlib.ticker
import model_learner as model
import numpy as np
import os.path as op
import pandas as pd
import plot_utils as plut
import scipy.stats

FIGSIZE = (plut.A4_PAPER_CONTENT_WIDTH * 1/3,
           plut.A4_PAPER_CONTENT_WIDTH * 1/3)
N_BINS_UNCERTAINTY = 6

def get_data_per_bin(x_vals, y_vals, bin_edges=None, n_bins=N_BINS_UNCERTAINTY):
    # Compute bins of x variable value by quantile cuts,
    # computed on the sequences seen by this subject
    if bin_edges is None:
        _, bin_edges = pd.qcut(x_vals, n_bins, retbins=True)
    n_bins = len(bin_edges) - 1
    # For each bin, compute the average x variable value and the
    # average+sem y variable value
    x_avg_per_bin = np.empty(n_bins)
    y_avg_per_bin = np.empty(n_bins)
    y_sem_per_bin = np.empty(n_bins)
    for i_bin in range(n_bins):
        filt = dana.get_bin_mask(x_vals, bin_edges, i_bin)
        x_avg_per_bin[i_bin] = np.mean(x_vals[filt])
        y_avg_per_bin[i_bin] = np.mean(y_vals[filt])
        y_sem_per_bin[i_bin] = scipy.stats.sem(y_vals[filt])
    return dict(x_avg=x_avg_per_bin,
        y_avg=y_avg_per_bin,
        y_sem=y_sem_per_bin)

def run_with_args(data_file, do_plot=True):
    data = dana.load_data(data_file)
    output_dir = dana.get_output_dir_for_data_file(data_file,
            create_if_needed=False)
    sessions_per_task = dana.get_sessions_per_task(data)
    n_tasks = len(sessions_per_task)
    for i_task, sessions in enumerate(sessions_per_task):
        retval = run_with_sessions(sessions,
            do_plot=do_plot,
            i_task=(i_task if n_tasks > 1 else None),
            output_dir=output_dir)
    return retval

def run_with_sessions(sessions,
    do_model=False, # do analysis on normative model's 
                    # learning rate instead of subjects'
    use_only_trials_with_update=False,
    do_plot=True,
    i_task=None,
    output_dir=None):
    fst_session = sessions[0]
    task = fst_session["taskName"]

    lr_key = "learning_rate"
    if do_model:
        lr_key = "model_learning_rate"
    array_keys = [lr_key, "model_uncertainty", "model_cpp"]
    if use_only_trials_with_update:
        array_keys += ["was_estimate_updated"]
    arrays = dana.get_data_arrays_from_sessions(sessions,
        array_keys)
    lrs = arrays[lr_key]
    uncerts = arrays["model_uncertainty"]
    cpps = arrays["model_cpp"]
    
    filt = np.ones_like(lrs, dtype=bool)
    filt &= ~np.isnan(lrs)
    if use_only_trials_with_update:
        filt &= arrays["was_estimate_updated"]

    lrs = lrs[filt].flatten()
    uncerts = uncerts[filt].flatten()
    cpps = cpps[filt].flatten()
    
    pearsonr_uncertainty, _ = scipy.stats.pearsonr(uncerts, lrs)
    pearsonr_cpp, _ = scipy.stats.pearsonr(cpps, lrs)

    retval = {
        "lrs": lrs,
        "uncertainty": uncerts,
        "cpp": cpps,
        "pearsonr_uncertainty": pearsonr_uncertainty,
        "pearsonr_cpp": pearsonr_cpp,
        }

    if not do_plot:
        return retval

    for var_values in [uncerts]:
        _, bin_edges = pd.qcut(uncerts, N_BINS_UNCERTAINTY, retbins=True)
        dana.create_dir_if_needed(output_dir)

        plut.setup_mpl_style()

        # Main effect of uncertainty on learning rate
        figname = f"learning-rate_by_uncertainty"
        if n_tasks > 1:
            figname += f"_task-{i_task+1}"
        if use_only_trials_with_update:
            figname += "_use-only-trials-with-update"
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.gca()
        xlabel = plut.get_uncertainty_label()
        ylabel = "Learning rate"
        ymin = min(0, np.percentile(lrs, 5))
        ymax = np.percentile(lrs, 95)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(ymin, ymax)
        ax.plot(var_values, lrs, '.', ms=1., color=plut.GRAY_COLOR, alpha=0.3)
        data_per_bin = get_data_per_bin(var_values, lrs, bin_edges=bin_edges)
        var_avg_per_bin = data_per_bin["x_avg"]
        lr_avg_per_bin = data_per_bin["y_avg"]
        lr_sem_per_bin = data_per_bin["y_sem"]
        for i_bin in range(N_BINS_UNCERTAINTY):
            ax.errorbar(var_avg_per_bin[i_bin],
                lr_avg_per_bin[i_bin],
                lr_sem_per_bin[i_bin],
                fmt='o', ms=1, color=plut.BLACK_COLOR)
        ax.text(0.05, 0.05, f"r={pearsonr_uncertainty:.2f}",
            ha="left", va="bottom",
            transform=ax.transAxes)
        figpath = dana.get_path(output_dir, figname)
        plut.save_figure(fig, figpath, do_fit_axis_labels=True)

    return retval

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file")
    args = parser.parse_args()
    data_file = args.data_file
    run_with_args(data_file)
