"""
Analyze learning rates locked around change points at the subject level.
"""

import argparse
import data_analysis_utils as dana
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import os.path as op
import plot_utils as plut
import scipy.stats

FNAME_PREFIX = "learning-rate_around_cp"

# Calculate figure sizes so that the graphs for the two tasks so that the space
# along the x axis separating each outcome is the same for the two tasks
FIGSIZE_W_PAD = 0.46
FIGSIZE_ADA_PROB = (
    ((plut.A4_PAPER_CONTENT_WIDTH - 2 * FIGSIZE_W_PAD)
        * dana.CP_DISTANCE_WINDOW_WIDTH_BY_TASK["ada-prob"]
        / (dana.CP_DISTANCE_WINDOW_WIDTH_BY_TASK["ada-prob"]
            + dana.CP_DISTANCE_WINDOW_WIDTH_BY_TASK["ada-pos"])
        + FIGSIZE_W_PAD),
    plut.DEFAULT_HEIGHT)
FIGSIZE_ADA_POS = (
    ((plut.A4_PAPER_CONTENT_WIDTH - 2 * FIGSIZE_W_PAD)
        * dana.CP_DISTANCE_WINDOW_WIDTH_BY_TASK["ada-pos"]
        / (dana.CP_DISTANCE_WINDOW_WIDTH_BY_TASK["ada-prob"]
            + dana.CP_DISTANCE_WINDOW_WIDTH_BY_TASK["ada-pos"])
        + FIGSIZE_W_PAD),
    plut.DEFAULT_HEIGHT)
XAXIS_HEIGHT = 120 / 300
STAT_HEIGHT = 0.10

def run_with_args(data_file, do_plot=True,
    use_only_trials_with_update=False):
    data = dana.load_data(data_file)
    output_dir = dana.get_output_dir_for_data_file(data_file,
        create_if_needed=True)
    sessions_per_task = dana.get_sessions_per_task(data)
    n_tasks = len(sessions_per_task)
    for i_task, sessions in enumerate(sessions_per_task):
        retval = run_with_sessions(sessions,
            use_only_trials_with_update=use_only_trials_with_update,
            i_task=(i_task if n_tasks > 1 else None),
            do_plot=do_plot,
            output_dir=output_dir)
    return retval

def run_with_sessions(sessions,
    use_only_trials_with_update=False,
    i_task=None,
    do_plot=True,
    do_ignore_start_cp=False,
    output_dir=None):
    window_around_cp = dana.get_window_around_cp_for_sessions(sessions)
    lrs_around_cp = dana.compute_lrs_around_cp(sessions, window_around_cp,
        use_only_trials_with_update=use_only_trials_with_update,
        do_ignore_start_cp=do_ignore_start_cp)
    retval = {
        "window_around_cp": window_around_cp,
        "lrs_around_cp": lrs_around_cp }
    if not do_plot:
        return retval

    lr_avg_around_cp = np.array([np.mean(lrs) for lrs in lrs_around_cp])
    lr_sem_around_cp = np.array([scipy.stats.sem(lrs) for lrs in lrs_around_cp])

    plut.setup_mpl_style()

    fig = make_figure(window_around_cp, lr_avg_around_cp, lr_sem_around_cp,
        use_only_trials_with_update=use_only_trials_with_update,
        task=sessions[0]["taskName"])
    figname = FNAME_PREFIX
    if i_task is not None:
        figname += f"_task-{i_task+1}"
    if use_only_trials_with_update:
        figname += "_use_only_trials_with_update"
    figpath = dana.get_path(output_dir, figname)
    plut.save_figure(fig, figpath)

    return retval

def make_figure(window_around_cp,
    lr_avg_around_cp,
    lr_sem_around_cp,
    task="ada-prob",
    clusters_sig_dists=None,
    use_only_trials_with_update=False,
    do_ignore_start_cp=False,
    color=plut.SUBJECT_COLOR,
    label=None):
    ylabel = "Learning rate"
    if task == "ada-prob":
        figsize = FIGSIZE_ADA_PROB
    elif task == "ada-pos":
        figsize = FIGSIZE_ADA_POS
    figsize = figsize[0], (figsize[1] + plut.TWIN_AXIS_PAD + XAXIS_HEIGHT)
    if clusters_sig_dists is not None:
        figsize = figsize[0], (figsize[1] + STAT_HEIGHT)
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    xvals = dana.get_outcomes_after_cp(window_around_cp)
    xlabel = plut.OUTCOMES_AFTER_CP_LABEL
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plot_lr_on_ax(ax, window_around_cp,
        lr_avg_around_cp,
        lr_sem_around_cp,
        color=color,
        label=label)

    # ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    xtick_offset = 2
    xtick_step = 2
    xticks = xvals[xtick_offset::xtick_step]
    ax.set_xticks(xticks)
    # twin axis to show time after change point below outcomes after change point
    ax_time = ax.twiny()
    twin_axis_offset_pt = (plut.TWIN_AXIS_PAD + XAXIS_HEIGHT) * 72
    ax_time.spines["bottom"].set_position(("outward", twin_axis_offset_pt))
    ax_time.xaxis.set_label_position("bottom")
    ax_time.xaxis.set_ticks_position("bottom")
    sec_after_cp = dana.get_sec_after_cp(window_around_cp)
    xtick_offset = 3
    xticks = xvals[xtick_offset::xtick_step]
    xticklabels = [f"{sec:.0f}" for sec in sec_after_cp[xtick_offset::xtick_step]]
    ax_time.set_xlim(ax.get_xlim())
    ax_time.set_xticks(xticks, labels=xticklabels)
    ax_time.set_xlabel(plut.SEC_AFTER_CP_LABEL)
    fig.sca(ax)

    texts = []
    if use_only_trials_with_update:
        texts = [plut.ONLY_TRIALS_WITH_UPDATE_LABEL]
    if do_ignore_start_cp:
        texts += ["ignore start c.p."]
    if len(texts) > 0:
        text = ", ".join(texts)
        ax.text(0.025, 1.0, text, ha="left", va="top",
                transform=ax.transAxes)

    # Plot statistically significant clusters of learning rates
    if clusters_sig_dists is not None:
        ylim = ax.get_ylim()
        ystar = ylim[1]
        for sig_dists in clusters_sig_dists:
            for s in sig_dists:
                ax.text(dana.get_outcomes_after_cp(s), ystar, '*',
                    ha="center", va="bottom")

    return fig

def plot_lr_on_ax(ax, window_around_cp,
    lr_avg_around_cp,
    lr_sem_around_cp,
    color=plut.SUBJECT_COLOR,
    label=None):
    xvals = dana.get_outcomes_after_cp(window_around_cp)
    ax.plot(xvals, lr_avg_around_cp, '-',
        label=label, color=color)
    if lr_sem_around_cp is not None:
        ax.fill_between(xvals,
            lr_avg_around_cp + lr_sem_around_cp,
            lr_avg_around_cp - lr_sem_around_cp,
            color=color, alpha=plut.ERROR_SHADING_ALPHA)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file")
    args = parser.parse_args()
    data_file = args.data_file
    run_with_args(data_file)
