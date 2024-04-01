"""
Plot the ranges of cpp and uncertainty values encountered in the two learning tasks.
"""

import data_analysis_utils as dana
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import plot_utils as plut

FNAME_PREFIX = "task_ranges"

def run_with_args():
    output_dir = dana.get_output_dir_for_group_def_file(dana.ADA_LEARN_GROUP_DEF_FILE,
        create_if_needed=True)
    dana.create_dir_if_needed(output_dir)

    # Compute change-point probability and prior uncertainty values for all sequences in each task
    data_by_task = {}
    keys = ["cpp", "uncertainty"]
    sessions_by_task = dana.make_sessions_by_task_from_seq_data(dana.ADA_LEARN_SEQ_DATA_FILE)
    for task, sessions in sessions_by_task.items():
        arrays = dana.get_data_arrays_from_sessions(sessions,
            keys=[f"model_{key}" for key in keys])
        data_by_task[task] = {
            key: arrays[f"model_{key}"].flatten() for key in keys
        }
    ranges = {key: {task: (data_by_task[task][key].min(), data_by_task[task][key].max()) 
        for task in data_by_task} for key in keys}
    medians = {key: {task: np.median(data_by_task[task][key]) 
        for task in data_by_task} for key in keys}

    plut.setup_mpl_style(fontsize=7)

    # Plot the value ranges of the two tasks separately for each variable
    for key in keys:
        FIGSIZE_RANGES = (plut.A4_PAPER_CONTENT_WIDTH * 1/4,
                          plut.A4_PAPER_CONTENT_WIDTH * 1/8)
        tasks = ["ada-pos", "ada-prob"]
        ylabels = ["Magnitude\nlearning", "Probability\nlearning"]
        yvals = np.arange(len(ylabels))
        xlabel = plut.get_reg_label(key, is_zscored=False, with_unit=False)
        xmins = np.array([ranges[key][t][0] for t in tasks])
        xmaxs = np.array([ranges[key][t][1] for t in tasks])
        xmeds = np.array([medians[key][t] for t in tasks])
        fig = plt.figure(figsize=FIGSIZE_RANGES)
        ax = fig.gca()
        ax.errorbar(xmeds, yvals,
            fmt="None",
            # fmt=".",
            yerr=None,
            xerr=((xmeds - xmins), (xmaxs - xmeds)),
            color=plut.BLACK_COLOR,
            ecolor=plut.BLACK_COLOR,
            elinewidth=1,
            capsize=3,
            capthick=1
            )
        ax.set_xlabel(xlabel)
        ax.set_yticks(yvals, ylabels)
        ax.set_ylim(yvals[0]-1/2, yvals[-1]+1/2)
        ax.invert_yaxis() # top to bottom
        figname = f"{FNAME_PREFIX}_{key}"
        for ext in ['png', 'pdf']:
            figpath = dana.get_path(output_dir, figname, ext=ext)
            plut.save_figure(fig, figpath)

    # Plot histogram of values separately for each task and variable
    maxranges = {key: 
        (min(data_by_task["ada-prob"][key].min(), data_by_task["ada-pos"][key].min()),
         max(data_by_task["ada-prob"][key].max(), data_by_task["ada-pos"][key].max()))
        for key in keys}
    for task, data in data_by_task.items():
        for key, values in data.items():
            FIGSIZE_HIST = (plut.A4_PAPER_CONTENT_WIDTH * 1/4,
                    plut.A4_PAPER_CONTENT_WIDTH * 1/4)
            N_BINS = 10
            xlabel = plut.get_reg_label(key, is_zscored=False)
            fig = plt.figure(figsize=FIGSIZE_HIST)
            ax = fig.gca()
            ax.set_xlabel(xlabel)
            ax.set_ylabel(f"Density")
            densities, bins, _ = ax.hist(values,
                bins=N_BINS,
                range=maxranges[key],
                color=plut.BLACK_COLOR,
                density=True)
            figname = f"{FNAME_PREFIX}_{key}_hist_{task}"
            for ext in ['png']:
                figpath = dana.get_path(output_dir, figname, ext=ext)
                plut.save_figure(fig, figpath)

if __name__ == '__main__':
    run_with_args()