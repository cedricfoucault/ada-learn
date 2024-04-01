"""
Plot illustration of adaptive vs fixed learning rate.
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

FNAME_PREFIX = "illustration_adaptability"

SEQ_IDX = {
    "ada-prob": 69,
    "ada-pos": 58,
}

FIXED_LEARNING_RATES = {
    "ada-prob": [0.05, 0.1, 0.35],
    "ada-pos": [0.25, 0.5, 0.95]
}

COLOR_LOW = "#E08585"
COLOR_MEDIUM = "#D14747"
COLOR_HIGH = "#A32929"

FIGSIZE_ILLUSTRATION_ADAPTABILITY = (2.49, 1.6)

def run_with_args():
    output_dir = dana.get_output_dir_for_group_def_file(
        group_def_file=dana.ADA_LEARN_GROUP_DEF_FILE,
                create_if_needed=True)
    sessions_by_task = dana.make_sessions_by_task_from_seq_data(
        seq_data_file=dana.ADA_LEARN_SEQ_DATA_FILE)

    for task in ["ada-prob", "ada-pos"]:
        seq_idx = SEQ_IDX[task]
        if task == "ada-prob":
            change_key = 'didP1Change'
        elif task == "ada-pos":
            change_key = 'didMeanChange'
        sessions = [sessions_by_task[task][seq_idx]]
        if task == "ada-prob":
            sessions[0]["outcome"][51] = 1 # change one outcome to
                                           # break the streak of 0s,
                                           # otherwise this example is too simplistic
        arrays = dana.get_data_arrays_from_sessions(sessions,
            keys=["model_estimate", "outcome", change_key])

        # TBD: DRY code to plot sessions with the code used in subject_sessions
        plut.setup_mpl_style(fontsize=7)
        outcomes = arrays["outcome"][0]
        model_estimate = arrays["model_estimate"][0]
        change_indices = np.argwhere(arrays[change_key][0])[1:]
            
        n_trials = outcomes.shape[0]
        trials = np.arange(1, n_trials + 1)
        trials_with_prior = np.array([0] + list(trials))
        lim_trials = trials[0] - 1, trials[-1] + 1
        label_trials = "Observations"
        estimate_trial_offset = 0.5
        lim_estimates = (-0.01, 1.01)
        if task == "ada-prob":
            label_estimates = "Probability p(blue)"
            # alpha_estimate = 0.8
            alpha_estimate = 0.9
        elif task == "ada-pos":
            label_estimates = "Position"
            # alpha_estimate = None
            # alpha_estimate = 0.8
            alpha_estimate = 0.9
        plot_args_s = []
        # Add outcomes in elements to plot
        if task == "ada-prob":
            for b in [0, 1]:
                color = (plut.ADA_PROB_OUTCOME_1_COLOR if b == 1
                    else plut.ADA_PROB_OUTCOME_0_COLOR)
                filt = (outcomes == b)
                plot_args_s += [dict(x=trials[filt], y=([0.5] * filt.sum()),
                    fmt='.',
                    # ms=2,
                    ms=1,
                    color=color)]
        elif task == "ada-pos":
            plot_args_s += [dict(x=trials, y=outcomes,
                    fmt='.', ms=1., color=plut.ADA_POS_OUTCOME_COLOR)]
        # Add fixed learning rate estimates
        # lw = 1.5
        lw = 1.
        for lr, lowhigh, color in zip(FIXED_LEARNING_RATES[task],
            ["low", "medium", "high"], [COLOR_LOW, COLOR_MEDIUM, COLOR_HIGH]):
            dr_estimate = model.delta_rule_estimates_from_outcomes(outcomes, lr)
            plot_args_s += [dict(x=trials_with_prior+estimate_trial_offset,
                y=dana.get_values_with_prior(dr_estimate, 0.5),
                fmt='-', color=color, alpha=alpha_estimate,
                label=f"fixed learning rate ({lowhigh})", lw=lw)]
        # Add model estimate
        plot_args_s += [dict(x=trials_with_prior+estimate_trial_offset,
            y=dana.get_values_with_prior(model_estimate, 0.5),
            fmt='-', color=plut.BLACK_COLOR, alpha=alpha_estimate,
            label=plut.IDEAL_LEARNER_LABEL, lw=lw)]

        fig = plt.figure(figsize=FIGSIZE_ILLUSTRATION_ADAPTABILITY)
        ax = fig.gca()
        xlim = lim_trials
        ylim = lim_estimates
        xlabel = label_trials
        ylabel = label_estimates
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # ax.set_xlabel(None)
        # ax.set_ylabel(None)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_xticks([0, n_trials])
        ax.set_yticks([0, 1])
        # Display change points
        for i in change_indices:
            ax.axvline(x=trials[i], ls=':', lw=1., color=plut.GENERATIVE_COLOR)
        for args in plot_args_s:
            x = args.pop("x")
            y = args.pop("y")
            fmt = args.pop("fmt")
            ax.plot(x, y, fmt, **args)

        figname = f"{FNAME_PREFIX}_{task}"
        for ext in ['png', 'pdf']:
            figpath = dana.get_path(output_dir, figname, ext)
            plut.save_figure(fig, figpath)

if __name__ == '__main__':
    run_with_args()

