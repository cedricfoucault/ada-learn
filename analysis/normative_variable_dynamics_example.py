"""
Plot the dynamics of the different normative model variables over the 
course of a sequence for an example sequence.
"""

import argparse
import data_analysis_utils as dana
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker
import model_learner as model
import numpy as np
import os.path as op
import sklearn.linear_model
import pandas as pd
import learning_rate_around_cp_subject
import learning_rate_relationship_cpp_uncertainty_subject
import plot_utils as plut
import scipy.stats

FNAME_PREFIX = "normative_variable_dynamics_example"

EXAMPLE_SESSION_IDX_ADA_POS = 5
EXAMPLE_SESSION_IDX_ADA_PROB = 3

PAPER_FIG_HEIGHT_EXAMPLE = 11.5 / plut.CM_PER_INCH
PAPER_FIG_WIDTH_EXAMPLE = plut.A4_PAPER_CONTENT_WIDTH / 2
PAPER_FONTSIZE = 7

def run_with_args(seq_data_file=dana.ADA_LEARN_SEQ_DATA_FILE):
    output_dir = dana.get_output_dir_for_group_def_file(dana.ADA_LEARN_GROUP_DEF_FILE,
        create_if_needed=True)

    sessions_by_task = dana.make_sessions_by_task_from_seq_data(seq_data_file)
    for task in ["ada-pos", "ada-prob"]:
        all_sessions = sessions_by_task[task]
        if task == "ada-prob":
            change_key = 'didP1Change'
        elif task == "ada-pos":
            change_key = 'didMeanChange'
        data = dana.get_data_arrays_from_sessions(all_sessions,
            keys=["nTrials", "outcome", change_key,
            "estimate", "learning_rate", "outcome-error-magnitude",
            "model_uncertainty", "model_cpp"])
        data["uncertainty"] = data["model_uncertainty"]
        data["cpp"] = data["model_cpp"]

        # Check spearman correlation between cpp and error magnitude
        err_magns = data["outcome-error-magnitude"].flatten()
        cpps = data["model_cpp"].flatten()
        spearman_res = scipy.stats.spearmanr(cpps, err_magns)
        print(f"Spearman correlation between change-point probability and error magnitude: {spearman_res}")

        plut.setup_mpl_style(fontsize=PAPER_FONTSIZE)
        label_by_key = {
            "learning_rate": "Learning rate",
            "outcome-error-magnitude": "Outcome error magnitude",
            "cpp": plut.CPP_LABEL,
            "uncertainty": plut.get_uncertainty_label()
        }

        #
        # Plot example session
        #
        outcome_surp_key = "cpp"
        if task == "ada-pos":
            example_session_indices = [EXAMPLE_SESSION_IDX_ADA_POS]
        elif task == "ada-prob":
            example_session_indices = [EXAMPLE_SESSION_IDX_ADA_PROB]
        for example_session_idx in example_session_indices:
            outcomes = data["outcome"][example_session_idx]
            estimates = data["estimate"][example_session_idx]
            learning_rates = data["learning_rate"][example_session_idx]
            outcome_vals = data[outcome_surp_key][example_session_idx]
            did_changes = data[change_key][example_session_idx]
            change_indices = np.argwhere(did_changes)[1:]
            uncertainties = data["uncertainty"][example_session_idx]
            estimates_with_prior = dana.get_values_with_prior(estimates, dana.ESTIMATE_PRIOR)

            figsize = (PAPER_FIG_WIDTH_EXAMPLE,
                       PAPER_FIG_HEIGHT_EXAMPLE
                       )
            fig, axes = plt.subplots(figsize=figsize, nrows=4, ncols=1)
            # Plot outcomes and estimates
            # TBD: DRY with subject_sessions
            n_trials = outcomes.shape[0]
            trials = np.arange(1, n_trials + 1)
            trials_with_prior = np.array([0] + list(trials))
            lim_estimates = (-0.01, 1.01)
            lim_trials = trials[0] - 1, trials[-1] + 1
            estimate_trial_offset = 0.5       
            ax = axes[0]
            xlim = lim_trials
            ylim = lim_estimates
            xlabel = "Observations"
            if task == "ada-prob":
                ylabel = "Probability p(blue)"
                alpha_estimate = 0.8
            elif task == "ada-pos":
                ylabel = "Position"
                alpha_estimate = None
            ax.set_ylabel(ylabel)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            for i in change_indices:
                ax.axvline(x=trials[i], ls=':', lw=1., color=plut.GENERATIVE_COLOR)
            if task == "ada-prob":
                for b in [0, 1]:
                    color = (plut.ADA_PROB_OUTCOME_1_COLOR if b == 1
                            else plut.ADA_PROB_OUTCOME_0_COLOR)
                    filt = (outcomes == b)
                    ax.plot(trials[filt], ([0.5] * filt.sum()),
                        '.', ms=2, color=color)
            elif task == "ada-pos":
                ax.plot(trials, outcomes, '.', ms=1, color=plut.ADA_POS_OUTCOME_COLOR)
            ax.plot(trials_with_prior + estimate_trial_offset,
                estimates_with_prior, "-", color=plut.BLACK_COLOR,
                alpha=alpha_estimate, label="Estimate")
            # ax.legend()
            # Plot cpp
            ax = axes[1]
            ylabel = label_by_key[outcome_surp_key]
            ax.set_ylabel(ylabel)
            ax.set_xlim(xlim)
            for i in change_indices:
                ax.axvline(x=trials[i], ls=':', lw=1., color=plut.GENERATIVE_COLOR)
            ax.plot(trials, outcome_vals, '-', color=plut.BLACK_COLOR)
            # Plot uncertainty
            ax = axes[2]
            ylabel = label_by_key["uncertainty"]
            ax.set_ylabel(ylabel)
            ax.set_xlim(xlim)
            for i in change_indices:
                ax.axvline(x=trials[i], ls=':', lw=1., color=plut.GENERATIVE_COLOR)
            ax.plot(trials, uncertainties, '-', color=plut.BLACK_COLOR)
            # Plot learning rate
            ax = axes[3]
            ylabel = label_by_key["learning_rate"]
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim(xlim)
            for i in change_indices:
                ax.axvline(x=trials[i], ls=':', lw=1., color=plut.GENERATIVE_COLOR)
            ax.plot(trials, learning_rates, '-', color=plut.BLACK_COLOR)

            figname = f"{FNAME_PREFIX}_{task}"
            for ext in ["png", "pdf"]:
                figpath = dana.get_path(output_dir, figname, ext)
                plut.save_figure(fig, figpath, do_fit_axis_labels=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    run_with_args()