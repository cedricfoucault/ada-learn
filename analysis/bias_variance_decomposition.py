"""
Do a decomposition analysis of the mean squared error between the subjects' estimates
and the normative model's estimates. The analysis is performed here at the group level.
The mean squared error represents the mean of the squared difference between
a subject's estimate and the normative model's estimate, across different
estimations, made by different subjects, given an identical stimulus sequence.

We are decomposing the mean squared error into two components:
(1) the squared bias error, which is the squared difference between
    the group mean estimate and the normative model's estimate,
    and quantifies systematic biases in the subjects' estimation process,
    which are reproducible across subjects;
(2) the variance, which is the mean of the squared difference between
    a subject's estimate and the group mean estimate, and quantifies the
    variability in subjects' estimation for the given sitmulus sequence.

Mathematically, the decomposition of the mean squared error is as follows
(ref. https://en.wikipedia.org/wiki/Mean_squared_error):

    mean-squared-error = squared-bias-error + variance

    mean-squared-error:
    mse =  1/N Σ_i (v^s(i) - v^n) ** 2

    squared-bias-error:
    sbe = (v^s - v^n) ** 2

    variance:
    var = 1/N Σ_i (v^s(i) - v^s) ** 2

    where:
    - v^n is the normative model's estimate for the given stimulus sequence
    - v^s(i) is the i-th sample of subject's estimate for the given stimulus sequence
            (i.e., the estimate of the i-th subject to which this stimulus sequence was shown)
    - v^s = 1/N Σ_i v^s(i) is the group mean estimate

The mse and its decomposition into sbe and variance are computed at the level of a trial,
and then at the level of a sequence by summing across trials.
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
import random
import scipy.stats

FNAME_PREFIX = "bias-variance_decomposition"
EXAMPLE_SESSION_IDX = {"ada-pos": 4, "ada-prob": 136}
VARIANCE_COLOR = "#376FB7"

def compute_mse_decomposition(subjects_estimate, model_estimate,
    do_sanity_check=True):
    """
    Compute the mean squared error, and its decomposition into squared bias error
    and variance, between the estimate of subjects and the estimate of a model.
    - subjects_estimate: np array of shape (n_subjects, *).
                         the first dimension should correspond to the different
                         estimations made by the different subjects who have
                         seen the considered stimulus sequence.
    - model_estimate: scalar or np array of shape (*).
                      the shapes of subjects_estimate[i] and model_estimate should match.

    Returns: dict containing the following key-value pairs:
    - 'mse': the mean squared error
    - 'sbe': the squared bias error
    - 'var': the variance
    """
    assert subjects_estimate.shape[0] > 1, \
        "need more then one sample to compute the mean/variance"
    assert subjects_estimate.ndim == model_estimate.ndim + 1
    # mean squared error
    se_subjects_model = (subjects_estimate - model_estimate[np.newaxis, :]) ** 2
    mse = np.mean(se_subjects_model, axis=0)
    # squared bias error
    groupmean_estimate = np.mean(subjects_estimate, axis=0)
    sbe = (groupmean_estimate - model_estimate) ** 2
    # variance
    # var = np.var(subjects_estimate, axis=0)
    se_subjects_groupmean = (subjects_estimate - groupmean_estimate) ** 2
    var = np.mean(se_subjects_groupmean, axis=0)
    if do_sanity_check:
        assert np.all(np.isclose(mse, sbe + var))
    # sum across the remaining dimensions (typically, the number of trials)
    mse = np.sum(mse)
    sbe = np.sum(sbe)
    var = np.sum(var)
    return {
        "mse": mse,
        "sbe": sbe,
        "var": var,
    }

def get_data_by_sequence_index(subjects_sessions):
    """Aggregate data for each unique stimulus sequence,
    iterating over subject and sessions and using the sequenceIdx
    to uniquely identify each stimulus sequence."""
    data_by_seq_idx = {}
    for i_sub, sessions in enumerate(subjects_sessions):
        arrays = dana.get_data_arrays_from_sessions(sessions,
            keys=["estimate", "model_estimate", "sequenceIdx", "outcome"])
        for i_sess in range(len(sessions)):
            seq_idx = arrays["sequenceIdx"][i_sess]
            if seq_idx not in data_by_seq_idx:
                data_by_seq_idx[seq_idx] = {
                    "outcome": arrays["outcome"][i_sess],
                    "model_estimate": arrays["model_estimate"][i_sess],
                    "subjects_estimate": [],
                }
            data_by_seq_idx[seq_idx]["subjects_estimate"] += [
                arrays["estimate"][i_sess]]
    # Convert the subjects' estimates for each sequence into a 2D numpy array
    # of shape (n_subjects, n_outcomes)
    for seq_idx, data in data_by_seq_idx.items():
        data["subjects_estimate"] = np.array(data["subjects_estimate"])
    return data_by_seq_idx

def fit_linear_bias_model(data_by_seq_idx, to="groupmean_estimate"):
    all_subject_estimates = []
    all_normative_model_estimates = []
    for seq_idx, data in data_by_seq_idx.items():
        subjects_estimate = data["subjects_estimate"]
        if to == "subjects_estimate":
            normative_model_estimates = np.repeat(
                data["model_estimate"][np.newaxis, :],
                repeats=subjects_estimate.shape[0],
                axis=0) # repeat the normative estimate for each subject
            all_subject_estimates += [subjects_estimate]
            all_normative_model_estimates += [normative_model_estimates]
        elif to == "groupmean_estimate":
            groupmean_estimate = np.mean(subjects_estimate, axis=0)
            normative_model_estimates = data["model_estimate"]
            all_subject_estimates += [groupmean_estimate]
            all_normative_model_estimates += [normative_model_estimates]
    all_subject_estimates = np.concatenate(all_subject_estimates).flatten()
    all_normative_model_estimates = np.concatenate(all_normative_model_estimates).flatten()
    linreg = scipy.stats.linregress(all_normative_model_estimates,
                                    all_subject_estimates)
    return {"intercept": linreg.intercept,
            "slope": linreg.slope}

def compute_linear_bias_model_estimate(linear_bias_model, normative_model_estimate):
    return (linear_bias_model["intercept"]
        + linear_bias_model["slope"] * normative_model_estimate)

def run_with_args(group_def_file):
    data_files = dana.get_subject_data_files_from_group_def(group_def_file)
    n_subjects = len(data_files)
    output_dir = dana.get_output_dir_for_group_def_file(group_def_file,
                create_if_needed=True)
    task_to_sub_sessions = dana.get_task_to_subject_sessions(data_files)
    n_tasks = len(task_to_sub_sessions)
    
    for task in ["ada-prob", "ada-pos"]:
        #
        # Decomposition of the mean squared error analysis
        #

        # Get data for each unique stimulus sequence
        data_by_seq_idx = get_data_by_sequence_index(task_to_sub_sessions[task])
        # Compute the mse decomposition for each stimulus sequence
        mse_decomp_by_seq_idx = { }
        for seq_idx, data in data_by_seq_idx.items():
            if (data["subjects_estimate"].shape[0] > 1):
                mse_decomp_by_seq_idx[seq_idx] = compute_mse_decomposition(
                    data["subjects_estimate"], data["model_estimate"])
                mse_decomp_by_seq_idx[seq_idx]["n_samples"] = data["subjects_estimate"].shape[0]
        n_effective_seqs = len(mse_decomp_by_seq_idx)

        # Compute the mse between subjects' estimate and the estimates of a
        # 'linear bias' model, rather than the normative model, in order to
        # how much of the mse with the normative model can be explained by such
        # a model of the bias.
        #
        # The 'linear bias' model applies a linear function to the normative model's estimates
        # A "conservatism bias" occurs when the slope of that linear function is < 1.
        #
        # The slope and intercept of that linear model are fitted to
        # the subjects' estimates at the group level.
        linear_bias_model = fit_linear_bias_model(data_by_seq_idx)
        for seq_idx, decomp in mse_decomp_by_seq_idx.items():
            data = data_by_seq_idx[seq_idx]
            normative_model_estimate = data["model_estimate"]
            linear_bias_model_estimate = compute_linear_bias_model_estimate(
                linear_bias_model, normative_model_estimate)
            decomp_linear_bias_model = compute_mse_decomposition(
                data["subjects_estimate"], linear_bias_model_estimate)
            decomp["mse_linear_bias_model"] = decomp_linear_bias_model["mse"]

        stat_data = {} # Object to store the outputs of this script

        stat_data["Number of sequences effectively considered in the analysis"] = \
            n_effective_seqs

        n_samples_per_seq = np.array([d["n_samples"] for d in mse_decomp_by_seq_idx.values()])
        stat_data["Median across sequences of the number of subject estimations per sequence"] = \
            np.median(n_samples_per_seq)
        stat_data["Mean across sequences of the number of subject estimations per sequence"] = \
            np.mean(n_samples_per_seq)
            
        # Compute the proportion of the total MSE across sequences
        # attributed to bias and variance.
        def compute_prop_sbe_total_mse_for_seq_indices(indices):
            total_mse = 0
            total_sbe = 0
            for seq_idx in indices:
                decomp = mse_decomp_by_seq_idx[seq_idx]
                total_mse += decomp["mse"]
                total_sbe += decomp["sbe"]
            prop_sbe = total_sbe / total_mse
            return prop_sbe
        seq_indices = [seq_idx for seq_idx in mse_decomp_by_seq_idx]
        prop_sbe_total_mse = compute_prop_sbe_total_mse_for_seq_indices(seq_indices)
        # Compute the proportion of the total MSE across sequences
        # that is explained by the linear-bias model
        def compute_prop_total_mse_explained_by_linear_bias_for_seq_indices(indices):
            total_mse = 0
            total_mse_linear_bias = 0
            for seq_idx in indices:
                decomp = mse_decomp_by_seq_idx[seq_idx]
                total_mse += decomp["mse"]
                total_mse_linear_bias += decomp["mse_linear_bias_model"]
            return (total_mse - total_mse_linear_bias) / total_mse
        prop_total_mse_explained_by_linear_bias = compute_prop_total_mse_explained_by_linear_bias_for_seq_indices(seq_indices)
        # Use a bootstrapping method to compute a s.e. of those estimates of
        # the proportion of total MSE, by resampling with replacement the original
        # set of sequences to create a new set of sequences.
        props_sbe_total_mse_bootstrap = []
        props_total_mse_explained_by_linear_bias_bootstrap = []
        n_sims_bootstrap = 10000
        random.seed(0)
        for i_sim in range(n_sims_bootstrap):
            # Randomly sample sequences with replacement from the original set of sequences
            resampled_seq_indices = random.choices(seq_indices, k=len(seq_indices))
            # Compute proportion for the new set of sequences obtained by resampling
            prop_sbe = compute_prop_sbe_total_mse_for_seq_indices(resampled_seq_indices)
            props_sbe_total_mse_bootstrap += [prop_sbe]
            prop_explained = compute_prop_total_mse_explained_by_linear_bias_for_seq_indices(resampled_seq_indices)
            props_total_mse_explained_by_linear_bias_bootstrap += [prop_explained]
        props_sbe_total_mse_bootstrap = np.array(props_sbe_total_mse_bootstrap)
        props_total_mse_explained_by_linear_bias_bootstrap = np.array(props_total_mse_explained_by_linear_bias_bootstrap)
        prop_sbe_total_mse_err = props_sbe_total_mse_bootstrap.std()
        prop_total_mse_explained_by_linear_bias_err = props_total_mse_explained_by_linear_bias_bootstrap.std()
        stat_data["Proportion of total MSE attributed to bias (%)"] = \
            prop_sbe_total_mse * 100
        stat_data["Proportion of total MSE attributed to variance (%)"] = \
            (1-prop_sbe_total_mse) * 100
        stat_data[("Standard error of the proportion of bias "
                   "estimated by bootstrapping (%)")] = \
            prop_sbe_total_mse_err * 100
        stat_data["Proportion of total MSE explained by the linear bias (%)"] = \
            prop_total_mse_explained_by_linear_bias * 100
        stat_data[("Standard error of the proportion of linear bias "
                   "estimated by bootstrapping (%)")] = \
            prop_total_mse_explained_by_linear_bias_err * 100
        # sanity check: the mean value over the bootstrap simulations should be
        # close to the value obtained with the original set of sequences
        assert np.isclose(np.mean(props_sbe_total_mse_bootstrap),
                          prop_sbe_total_mse,
                          atol=0.001)
        assert np.isclose(np.mean(props_total_mse_explained_by_linear_bias_bootstrap),
                          prop_total_mse_explained_by_linear_bias,
                          atol=0.001)

        # Save results in CSV
        stat_data = pd.Series(stat_data)
        stat_name = f"{FNAME_PREFIX}_{task}"
        stat_fpath = dana.get_path(output_dir, stat_name, 'csv')
        plut.save_stats(stat_data, stat_fpath)

        # Plot results of the decomposition of the MSE as a bar diagram
        plut.setup_mpl_style()
        pct_sbe = prop_sbe_total_mse * 100
        pct_var = (1-prop_sbe_total_mse) * 100
        pct_sbe_err = prop_sbe_total_mse_err * 100
        figsize = (2.96, plut.DEFAULT_HEIGHT / 2)
        bar_height = 20 / 65.2
        bar_bottom = 0.05
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        ax.spines[["left", "right"]].set_visible(True)
        ax.spines[["top", "bottom"]].set_visible(False)
        bar = ax.barh([bar_bottom], [pct_sbe], height=bar_height, align='edge',
            label="bias", left=0,
            color=plut.IDEAL_LEARNER_COLOR, edgecolor=plut.BLACK_COLOR, lw=1)
        ax.bar_label(bar, fmt="%.1f%%", label_type='center',
            color=plut.BLACK_COLOR)
        bar = ax.barh([bar_bottom], [pct_var], height=bar_height, align='edge',
            label="variance", left=pct_sbe,
            color=VARIANCE_COLOR,
            edgecolor=plut.BLACK_COLOR, lw=1)
        ax.bar_label(bar, fmt="%.1f%%", label_type='center',
            color=plut.BLACK_COLOR)
        ax.plot([pct_sbe-pct_sbe_err, pct_sbe+pct_sbe_err],
            [bar_bottom+bar_height/2, bar_bottom+bar_height/2],
            '-', color=plut.BLACK_COLOR, lw=1) # error bar
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        figname = f"{FNAME_PREFIX}_{task}"
        for ext in ['png', 'pdf']:
            figpath = dana.get_path(output_dir, figname, ext)
            plut.save_figure(fig, figpath)

        #
        # Example sequence plot
        #

        #
        # Plot illustration for an example sequence, showing the estimates
        # for all subjects, their mean estimate and the normative estimates
        # along with the sequence
        #
        seq_idx = EXAMPLE_SESSION_IDX[task]
        data = data_by_seq_idx[seq_idx]
        outcomes = data["outcome"]
        model_estimate = data["model_estimate"]
        subjects_estimate = data["subjects_estimate"]
        groupmean_estimate = np.mean(subjects_estimate, axis=0)
        # TBD: DRY code to plot sequence with the code used in subject_sessions
        n_trials = outcomes.shape[0]
        trials = np.arange(1, n_trials + 1)
        trials_with_prior = np.array([0] + list(trials))
        lim_trials = trials[0] - 1, trials[-1] + 1
        label_trials = "Observations"
        estimate_trial_offset = 0.5
        lim_estimates = (-0.01, 1.01)
        if task == "ada-prob":
            label_estimates = "Probability p(blue)"
            alpha_estimate = 0.8
        elif task == "ada-pos":
            label_estimates = "Position"
            # alpha_estimate = None
            alpha_estimate = 0.8
        plot_args_s = []
        # Add outcomes in elements to plot
        if task == "ada-prob":
            for b in [0, 1]:
                color = (plut.ADA_PROB_OUTCOME_1_COLOR if b == 1
                    else plut.ADA_PROB_OUTCOME_0_COLOR)
                filt = (outcomes == b)
                plot_args_s += [dict(x=trials[filt], y=([0.5] * filt.sum()),
                    fmt='.', ms=2, color=color)]
        elif task == "ada-pos":
            plot_args_s += [dict(x=trials, y=outcomes,
                    fmt='.', ms=1., color=plut.ADA_POS_OUTCOME_COLOR)]
        # Add model estimate
        plot_args_s += [dict(x=trials_with_prior+estimate_trial_offset,
            y=dana.get_values_with_prior(model_estimate, 0.5),
            fmt='-', color=plut.IDEAL_LEARNER_COLOR, alpha=alpha_estimate,
            label=plut.IDEAL_LEARNER_LABEL, lw=1.5)]
        # Add group mean estimate
        plot_args_s += [dict(x=trials_with_prior+estimate_trial_offset,
            y=dana.get_values_with_prior(groupmean_estimate, 0.5),
            fmt='-', color=plut.SUBJECT_COLOR, alpha=alpha_estimate,
            label=plut.SUBJECT_LABEL, lw=1.5)]
        # Add subjects' estimates
        for estimate in subjects_estimate:
            plot_args_s += [dict(x=trials_with_prior+estimate_trial_offset,
                y=dana.get_values_with_prior(estimate, 0.5),
                fmt='-', color=plut.COLORS_PRIMARY_BLUE[4], alpha=1/4,
                lw=1/2)]
        figsize = (plut.A4_PAPER_CONTENT_WIDTH * 1/2, plut.DEFAULT_HEIGHT)
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        xlim = lim_trials
        ylim = lim_estimates
        xlabel = label_trials
        ylabel = label_estimates
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for args in plot_args_s:
            x = args.pop("x")
            y = args.pop("y")
            fmt = args.pop("fmt")
            ax.plot(x, y, fmt, **args)
        figname = f"{FNAME_PREFIX}_illustration-sequence_{task}"
        for ext in ['png', 'pdf']:
            figpath = dana.get_path(output_dir, figname, ext)
            plut.save_figure(fig, figpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group_def_file", type=str,
        default=dana.ADA_LEARN_GROUP_DEF_FILE)
    args = parser.parse_args()
    run_with_args(args.group_def_file)
