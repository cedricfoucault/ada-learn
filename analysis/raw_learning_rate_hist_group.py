"""
Plot histogram of raw learning rates for all trials in a group dataset.
"""

import argparse
import data_analysis_utils as dana
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import plot_utils as plut
import scipy.stats

FIGSIZE = (plut.A4_PAPER_CONTENT_WIDTH * 1/2, plut.DEFAULT_HEIGHT)

SMALL_BIN_WIDTH_LR = 0.1
SMALL_BIN_MIN_LR = -3 - (1e-10) # - epsilon so that the bin that contain 0 has 0 at its left edge
SMALL_BIN_MAX_LR = +4.
HIST_MIN_LR = SMALL_BIN_MIN_LR -0.5
HIST_MAX_LR = SMALL_BIN_MAX_LR +0.5

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group_def_file")
    args = parser.parse_args()
    group_def_file = args.group_def_file

    data_files = dana.get_subject_data_files_from_group_def(group_def_file)
    output_dir = dana.get_output_dir_for_group_def_file(group_def_file,
        create_if_needed=True)

    n_subjects = len(data_files)
    # compute raw learning rates for all trials in the dataset
    raw_lrs = []
    for data_file in data_files:
        data = dana.load_data(data_file)
        sessions = data["taskSessionDataArray"]
        measures = dana.estimate_update_measures_from_sessions(sessions)
        raw_lrs += measures["learning_rates_raw"].flatten().tolist()
    raw_lrs = np.array(raw_lrs)
    raw_lr_min = raw_lrs.min()
    raw_lr_max = raw_lrs.max()

    bins = [raw_lr_min]
    edge = SMALL_BIN_MIN_LR
    i_bin = 1
    while edge <= SMALL_BIN_MAX_LR:
        if np.isclose(edge, 0):
            i_bin_0 = i_bin
        elif np.isclose(edge, 1):
            i_bin_1 = i_bin
        bins += [edge]
        edge += SMALL_BIN_WIDTH_LR
        i_bin += 1
    if raw_lr_max >= SMALL_BIN_MAX_LR:
        bins += [raw_lr_max]

    plut.setup_mpl_style()

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.gca()
    ax.set_xlabel("Learning rate (raw)")
    ax.set_ylabel(f"Density")
    densities, bins, _ = ax.hist(raw_lrs,
        bins=bins,
        color=plut.SUBJECT_COLOR,
        density=True)
    ax.set_xlim(HIST_MIN_LR, HIST_MAX_LR)
    # ax.text(0.05, 0.95, f"N={n_subjects}", ha="left", va="top",
    #     transform=ax.transAxes)
    arrowprops = dict(width=1., headwidth=2., headlength=2.)
    ax.annotate(f"min={raw_lr_min:.0f}", xy=(0.01, 0.1), xytext=(0.06, 0.1),
        xycoords=ax.transAxes, textcoords=ax.transAxes,
        ha="left", va="center", arrowprops=arrowprops,
        fontsize=7)
    ax.annotate(f"max={raw_lr_max:.0f}", xy=(0.99, 0.1), xytext=(0.94, 0.1),
        xycoords=ax.transAxes, textcoords=ax.transAxes,
        ha="right", va="center", arrowprops=arrowprops,
        fontsize=7)

    # compute and show lr min and max below/above which the density is below that
    # of a normal distribution at z=2
    density_z_2 = scipy.stats.norm.pdf(2)
    i_bin_left = i_bin_0 + 1
    i_bin_right = i_bin_1 - 1
    while densities[i_bin_left-1] >= density_z_2:
        i_bin_left -= 1
    while densities[i_bin_right] >= density_z_2:
        i_bin_right += 1
    ax.axhline(y=density_z_2, ls='--', lw=1, label=f"z=2", color=plut.GRAY_COLOR)
    ax.axvline(x=bins[i_bin_left], ymin=0, ymax=1/densities.max(), ls='--', lw=1, color=plut.GRAY_COLOR)
    ax.axvline(x=bins[i_bin_right], ymin=0, ymax=1/densities.max(), ls='--', lw=1, color=plut.GRAY_COLOR)
    ax.text(-1.5, density_z_2*1.01, f"density(z=2)", fontsize=7, va="bottom", ha="center")
    ax.text(bins[i_bin_left], 1.05, f"{bins[i_bin_left]:.2f}", fontsize=7, va="bottom", ha="center")
    ax.text(bins[i_bin_right], 1.05, f"{bins[i_bin_right]:.2f}", fontsize=7, va="bottom", ha="center")

    figname = f"raw-learning-rate-histogram.png"
    figpath = op.join(output_dir, figname)
    plut.save_figure(fig, figpath)
