"""
Module for defining and computing different behavioural measures of one subject in one task.
"""

import argparse
import data_analysis_utils as dana
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.linear_model

BMEAS_KEYS = [
    "update-freq",
    "avg-lr",
    "baseline-lr",
    "diff-lr-close-far-cp",
    "diff-lr-close-baseline-cp",
    "diff-lr-close-before-cp",
    "diff-lr-high-low-uncertainty",
    "diff-lr-high-low-cpp",
    "corr-lr-uncertainty",
    "corr-lr-cpp",
    "coef-z-uncertainty",
    "coef-z-cpp",
    "coef-z-uncertainty-control",
    "coef-z-cpp-control",
    "rt"
]

WINDOW_BOUNDS_BY_TASK = {
    "ada-prob": {
        # in this task, the median num. of trials between two change points is 16 (mean 19).
        # close corresponds the first half median duration.
        "close": [0, 7],
        "far": [8, 15],
        "before": [-2, -1],
    },
    "ada-pos": {
        # in this task, the median num. of trials between two change points is 8 (mean 10.6).
        # close corresponds the first half median duration.
        "close": [0, 3],
        "far": [4, 8], # [4, 9]
        "before": [-2, -1],
    }
}

def compute_behavioral_measures(sessions,
    do_rt=False, sessions_event_segments=None):
    """
    Returns a dictionary with the following scalar measures.
    - "update-freq":
        average update frequency (how often they updated their estimate).
    - "avg-lr":
        average learning rate across all trials.
    - "baseline-lr":
        average learning rate in the baseline trials,
        i.e. the q-furthest trials after c.p. (q=20%)
    - "diff-lr-close-far-cp":
        diff in average learning rate in trials close - far after c.p.
    - "diff-lr-close-baseline-cp":
        diff in average learning rate in trials close after c.p. - baseline trials 
    - "diff-lr-close-before-cp":
        diff in average learning rate in trials close after c.p. - just before c.p.
    - "diff-lr-high-low-cpp":
        diff in average learning rate in trials split by higher/lower cpp
    - "diff-lr-high-low-uncertainty":
        diff in average learning rate in trials split by higher/lower uncertainty
    - "corr-lr-cpp":
        trial-wise correlation between learning rate and cpp
    - "corr-lr-uncertainty":
        trial-wise correlation between learning rate and uncertainty
    - "coef-z-cpp":
        beta coefficient of z-scored cpp in the multiple linear regression
        on learning rate
    - "coef-z-uncertainty":
        beta coefficient of z-scored uncertainty in the multiple linear regression
        on learning rate,
    - "corr-lr-model":
        trial-wise correlation between subject's and model's learning rate
    [TBD: Add explanation for
    -"coef-z-uncertainty-control",
    -"coef-z-cpp-control",
    ]

    if do_rt is True sessions_event_segments are provided:
        - "rt":
            mean response latency
    """
    n_sessions = len(sessions)
    task = sessions[0]["taskName"]

    keys = ["nTrials", "was_estimate_updated", "learning_rate",
        "model_uncertainty", "model_cpp", "model_learning_rate",
        "estimate", "model_estimate"]
    arrays = dana.get_data_arrays_from_sessions(sessions, keys)

    lrs = arrays["learning_rate"]

    measures = {}

    measures["update-freq"] = np.mean(arrays["was_estimate_updated"])

    measures["avg-lr"] = np.nanmean(lrs)

    baseline_trials = dana.compute_baseline_trials(sessions,
        method="quantile_across_sessions", q_baseline=0.2)
    lrs_baseline = dana.aggregate_values_in_baseline_trials(lrs, baseline_trials)
    measures["baseline-lr"] = np.mean(lrs_baseline)

    window_around_cp = dana.get_window_around_cp(task)
    lrs_around_cp = dana.aggregate_values_in_window_around_cp(lrs,
        window_around_cp, sessions)
    lr_around_cp = np.array([np.mean(lrs) for lrs in lrs_around_cp])
    for (meas_key, pos_bounds_key, neg_bounds_key) in [
        ("diff-lr-close-far-cp", "close", "far"),
        ("diff-lr-close-baseline-cp", "close", "baseline"),
        ("diff-lr-close-before-cp", "close", "before")]:
        pos_bounds = WINDOW_BOUNDS_BY_TASK[task][pos_bounds_key]
        pos_mask = ((window_around_cp >= pos_bounds[0])
            & (window_around_cp <= pos_bounds[1]))
        pos_lr = np.nanmean(lr_around_cp[pos_mask])
        if neg_bounds_key == "baseline":
            neg_lr = measures["baseline-lr"]
        else:
            neg_bounds = WINDOW_BOUNDS_BY_TASK[task][neg_bounds_key]
            neg_mask = ((window_around_cp >= neg_bounds[0])
                & (window_around_cp <= neg_bounds[1]))
            neg_lr = np.nanmean(lr_around_cp[neg_mask])
        measures[meas_key] = (pos_lr - neg_lr)

    filt = ~np.isnan(lrs)

    z_cpp = scipy.stats.zscore(arrays["model_cpp"][filt])
    z_uncert = scipy.stats.zscore(arrays["model_uncertainty"][filt])
    z_lr = scipy.stats.zscore(lrs[filt])
    X = np.column_stack([z_cpp, z_uncert])
    linreg = sklearn.linear_model.LinearRegression(fit_intercept=True).fit(X, z_lr)

    arrays["model_uncertainty*(1-cpp)"] = arrays["model_uncertainty"] * (1 - arrays["model_cpp"])
    z_uncert_control_cpp = scipy.stats.zscore(arrays["model_uncertainty*(1-cpp)"][filt])
    X_control = np.column_stack([z_cpp, z_uncert_control_cpp])
    linreg_control = sklearn.linear_model.LinearRegression(fit_intercept=True).fit(X_control, z_lr)

    for i_var, var_key in enumerate(["cpp", "uncertainty"]):
        # z-coefficients of variables in linear regression on learning rate controlling for the other variable
        measures[f"coef-z-{var_key}-control"] = linreg_control.coef_[i_var]
        # z-coefficients of variables in linear regression on learning rate
        measures[f"coef-z-{var_key}"] = linreg.coef_[i_var]
        # Pearson correlation of individual variable with learning rate
        var_vals = arrays["model_" + var_key]
        corr, _ = scipy.stats.pearsonr(var_vals[filt], lrs[filt])
        measures[f"corr-lr-{var_key}"] = corr
        # Difference in mean learning rate across trials split by low-high value of variable
        if (task == "ada-pos") and ("var_key" == "cpp"):
            bin_edges = np.linspace(var_vals.min(), var_vals.max(), num=3)
        else:
            _, bin_edges = pd.qcut(var_vals.flatten(), 2, retbins=True)
        diff_lr_high_low = 0
        for i_bin in range(2):
            bin_mask = dana.get_bin_mask(var_vals, bin_edges, i_bin)
            lr = np.nanmean(lrs[bin_mask])
            diff_lr_high_low += -lr if (i_bin == 0) else +lr
        measures[f"diff-lr-high-low-{var_key}"] = diff_lr_high_low

    measures["corr-lr-model"] = scipy.stats.pearsonr(lrs[filt],
        arrays["model_learning_rate"][filt])[0]

    measures["corr-estimate-model"] = scipy.stats.pearsonr(arrays["estimate"].reshape(-1),
        arrays["model_estimate"].reshape(-1))[0]

    if do_rt and (sessions_event_segments is not None):
        rts = dana.compute_response_latencies(sessions, sessions_event_segments)
        measures["rt"] = np.nanmean(rts)

    return measures

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group_def_file")
    args = parser.parse_args()

    group_def_file = args.group_def_file
    data_files = dana.get_subject_data_files_from_group_def(group_def_file)
    n_subjects = len(data_files)

    data = dana.load_data(data_files[0])
    n_tasks = data.get("nTasks", 1)
    if n_tasks > 1:
        tasks = [sessions[0]["taskName"] for sessions in data["taskSessionDataArrays"]]
    else:
        tasks = [data["taskSessionDataArray"][0]["taskName"]]

    for i_sub, data_file in enumerate(data_files):
        data = dana.load_data(data_file)
        for task in tasks:
            if n_tasks > 1:
                for task_sessions in data["taskSessionDataArrays"]:
                    if task_sessions[0]["taskName"] == task:
                        sessions = task_sessions
                        break
            else:
                sessions = data["taskSessionDataArray"]

            measures = compute_behavioral_measures(sessions)
            print("i_sub, task", i_sub, task)
            print(measures)
    


    