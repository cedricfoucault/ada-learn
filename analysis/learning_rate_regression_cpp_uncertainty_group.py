"""
Perform a multiple linear regression of model variables (cpp, uncertainty)
on subject's learning rate.
"""

import argparse
import data_analysis_utils as dana
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import sklearn.linear_model
import pandas as pd
import plot_utils as plut
import scipy.stats

FNAME_PREFIX = "learning-rate_regression_weights_cpp_uncertainty"

STAT_CMP_OFFSET_FRAC = 0.05
STAT_HEIGHT = 0.10
XAXIS_HEIGHT = 110 / 300

def coef_key_for_var_key(key, do_control=False):
    coef_key = "coef_" + key
    if do_control:
        coef_key += "-control"
    return coef_key

def get_uncert_key(do_control=False):
    if do_control:
        return "relative-uncertainty-*-1-cpp"
    else:
        return "uncertainty"

def get_regvar_keys(do_control=False):
    return ["cpp", get_uncert_key(do_control)]

def get_coef_keys(do_control=False):
    return [coef_key_for_var_key(k, do_control=do_control)
        for k in get_regvar_keys(do_control=do_control)]

def compute_reg_weights_per_sub(sub_sessions,
    estimate_key="estimate",
    estimate_funs=None,
    use_only_trials_with_update=False,
    do_control=False):
    # Compute regression weights for each subject
    n_subjects = len(sub_sessions)
    uncert_key = get_uncert_key(do_control)
    reg_weights_per_sub = { }
    for k in ["cpp", uncert_key]:
        coef_key = coef_key_for_var_key(k, do_control=do_control)
        reg_weights_per_sub[coef_key] = np.empty(n_subjects)
    for i_sub, sessions in enumerate(sub_sessions):
        model_uncert_key = ("model_relative-uncertainty" if do_control
            else "model_uncertainty")
        array_keys = ["outcome", model_uncert_key, "model_cpp"]
        if use_only_trials_with_update:
            array_keys += ["was_estimate_updated"]
        if estimate_funs is None:
            array_keys += [estimate_key]
        arrays = dana.get_data_arrays_from_sessions(sessions, array_keys)
        outcome = arrays["outcome"]
        if estimate_funs is None:
            estimate = arrays[estimate_key]
        else:
            estimate = estimate_funs[i_sub](outcome)
        lrs = dana.estimate_update_measures(estimate, outcome)["learning_rates"]
        filt = ~np.isnan(lrs)
        if use_only_trials_with_update:
            filt = filt & arrays["was_estimate_updated"]
        lrs = lrs[filt]
        cpps = arrays["model_cpp"][filt]
        uncerts = arrays[model_uncert_key][filt]
        if do_control:
            uncerts = uncerts * (1 - cpps)
        linreg_vars = pd.DataFrame({
            "learning_rate": lrs,
            uncert_key: uncerts,
            "cpp": cpps,
            })
        if (not do_control):
            # zscore before running linear regression
            for k, x in linreg_vars.items():
                linreg_vars[k] = scipy.stats.zscore(x)
        x_cols = [uncert_key, "cpp"]
        X = linreg_vars[x_cols].to_numpy()
        y = linreg_vars["learning_rate"].to_numpy()
        linreg = sklearn.linear_model.LinearRegression(fit_intercept=True).fit(X, y)
        for i, k in enumerate(x_cols):
            coef_key = coef_key_for_var_key(k, do_control=do_control)
            reg_weights_per_sub[coef_key][i_sub] = linreg.coef_[i]
    return reg_weights_per_sub


def compute_stats_on_reg_weights(reg_weights_per_sub,
    do_control=False):
    coef_keys = get_coef_keys(do_control=do_control)
    stat_data = {}
    for coef_key in coef_keys:
        coefs = reg_weights_per_sub[coef_key]
        group_mean = coefs.mean()
        group_sem = scipy.stats.sem(coefs)
        stat_data[f"{coef_key}_mean"] = group_mean
        stat_data[f"{coef_key}_sem"] = group_sem
        # stat test coef > 0
        testres = scipy.stats.ttest_1samp(coefs, 0)
        stat_data[f"{coef_key}_t-test_t"] = testres.statistic
        stat_data[f"{coef_key}_t-test_p"] = testres.pvalue
    # stat test coef_cpp != coef_uncert
    coefs_pair = [reg_weights_per_sub[coef_key] for coef_key in coef_keys]
    testres = scipy.stats.ttest_rel(coefs_pair[0], coefs_pair[1],
        alternative='two-sided')
    stat_data[f"coef-diff_t-test_t"] = testres.statistic
    stat_data[f"coef-diff_t-test_p"] = testres.pvalue
    return stat_data

def plot_on_ax(ax, stat_data, show_err=True, show_sig=True, small_label=False,
    use_only_trials_with_update=False,
    do_control=False,
    color=plut.BLACK_COLOR,
    ecolor=plut.GRAY_COLOR):
    regvar_keys = get_regvar_keys(do_control=do_control)
    x_labels = [plut.get_reg_label(k,
        is_zscored=True,
        is_small_width=small_label)
            for k in regvar_keys]
    for i, k in enumerate(regvar_keys):
        coef_key = coef_key_for_var_key(k, do_control=do_control)
        group_mean = stat_data[f"{coef_key}_mean"]
        if show_err:
            group_sem = stat_data[f"{coef_key}_sem"]
            ax.bar(i, group_mean,
                width=0.5,
                yerr=group_sem,
                color=color,
                ecolor=ecolor)
        else:
            ax.bar(i, group_mean,
                width=0.5,
                color=color)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(plut.REG_COEF_LABEL)
    if show_sig:
        # add stat test coef > 0
        for i, k in enumerate(regvar_keys):
            coef_key = coef_key_for_var_key(k, do_control=do_control)
            stars = plut.stat_label(stat_data[f"{coef_key}_t-test_p"])
            ylim = ax.get_ylim()
            text_y = ylim[0] + (ylim[1] - ylim[0]) * 0.95
            ax.text(i, text_y,
                stars,
                va="bottom", ha="center")
        # add stat test coef_1 != coef_2 
        stars = plut.stat_label(stat_data[f"coef-diff_t-test_p"])
        ax.text(0.5, (1+STAT_CMP_OFFSET_FRAC), stars,
            va="bottom", ha="center",
            transform=ax.transAxes)
    if use_only_trials_with_update:
        ax.text(0.025, 1.0, plut.ONLY_TRIALS_WITH_UPDATE_LABEL, ha="left", va="top",
            transform=ax.transAxes)

def run_with_args(group_def_file,
    do_model=False,
    use_only_trials_with_update=False,
    do_control=False,
    small=False):
    """
    Parameters:
    - do_model:
        If True, compute the results predicted by the normative model,
        doing the analysis in exactly the same way as for subjects
        but using the model's learning rate rather than the subject's.
    - use_only_trials_with_update:
        If True, all data points where the subject did not perform an update
        (yielding a learning rate equal to 0) will be discarded from the analysis.
    - do_control:
        Only applies for the ada-pos task.
        Rather than performing a regression analysis using the
        prior uncertainty and cpp regressors and having z-scored all variables,
        perform another regression as in McGuire et al. (2014),
        for comparison.
    """
    data_files = dana.get_subject_data_files_from_group_def(group_def_file)
    n_subjects = len(data_files)
    task_to_sub_sessions = dana.get_task_to_subject_sessions(data_files)
    n_tasks = len(task_to_sub_sessions)
    output_dir = dana.get_output_dir_for_group_def_file(group_def_file,
                create_if_needed=True)

    for task, sub_sessions in task_to_sub_sessions.items():
        if do_control and task == "ada-prob":
            continue # control analysis not implemented for ada-prob

        sub_sessions = task_to_sub_sessions[task]
        estimate_key = "estimate"
        if do_model:
            estimate_key = "model_estimate"
        subs_data = compute_reg_weights_per_sub(sub_sessions,
            estimate_key=estimate_key,
            use_only_trials_with_update=use_only_trials_with_update,
            do_control=do_control)
        retval = subs_data

        coef_keys = get_coef_keys(do_control=do_control)

        def get_fname(prefix):
            fname = prefix
            if use_only_trials_with_update:
                fname += "_use-only-trials-with-update"
            if small:
                fname += "_small"
            if do_model:
                fname += "_model"
            if n_tasks > 1:
                fname += f"_{task}"
            return fname

        if do_control:
            # For the control analysis, compute the effect size and stats
            # as this was done in McGuire et al., 2014, for comparison.
            stat_data = {}
            for coef_key in coef_keys:
                coefs = subs_data[coef_key]
                median = np.median(coefs)
                q1 = np.quantile(coefs, 0.25)
                q3 = np.quantile(coefs, 0.75)
                iqr = q3 - q1
                wilcox = scipy.stats.wilcoxon(coefs)
                stat_data[f"{coef_key}_median"] = median
                stat_data[f"{coef_key}_Q1"] = q1
                stat_data[f"{coef_key}_Q3"] = q3
                stat_data[f"{coef_key}_IQR"] = iqr
                stat_data[f"{coef_key}_signed-rank_p"] = wilcox.pvalue
            stat_data = pd.Series(stat_data)
            stat_fname = get_fname(FNAME_PREFIX + "_control-as-in-mcguire")
            stat_fpath = dana.get_path(output_dir, stat_fname, 'csv')
            stat_data.to_csv(stat_fpath)
            print(f"Saved stats at {stat_fpath}")
            plut.save_stats(stat_data, stat_fpath)
            return retval

        # Compute stats
        stat_data = compute_stats_on_reg_weights(subs_data,
            do_control=do_control)

        # Plot and save stats
        if do_model or small:
            plut.setup_mpl_style(fontsize=7)
        else:
            plut.setup_mpl_style()
        if do_model:
            figsize = (4.2 / plut.CM_PER_INCH,
                       3.48 / plut.CM_PER_INCH)

        else:
            figsize = (plut.A4_PAPER_CONTENT_WIDTH / 3,
                       plut.A4_PAPER_CONTENT_WIDTH / 3)
            if small:
                figsize = (plut.A4_PAPER_CONTENT_WIDTH / 4,
                           plut.A4_PAPER_CONTENT_WIDTH / 4 - 0.5)
            figheight = (figsize[1]
                    + (figsize[1] - XAXIS_HEIGHT) * (STAT_CMP_OFFSET_FRAC)
                    + STAT_HEIGHT)
            figsize = (figsize[0], figheight)
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        plot_on_ax(ax, stat_data,
            show_err=(not do_model),
            show_sig=(not do_model),
            small_label=(small or do_model),
            use_only_trials_with_update=use_only_trials_with_update,
            do_control=do_control,
            color=(plut.COLORS_PRIMARY_BLUE[7] if not do_model else plut.BLACK_COLOR))
        if do_model:
            ax.set_yticks([0, 0.4, 0.8])

        fname = get_fname(FNAME_PREFIX)
        for ext in ["png", "pdf"]:
            figpath = dana.get_path(output_dir, fname, ext)
            plut.save_figure(fig, figpath, do_fit_axis_labels=True)
        
        if (not do_model):
            stat_data = pd.Series(stat_data)
            stat_fname = get_fname(FNAME_PREFIX + "_stats")
            stat_fpath = dana.get_path(output_dir, stat_fname, 'csv')
            plut.save_stats(stat_data, stat_fpath)

    return retval

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group_def_file", type=str,
        default=dana.ADA_LEARN_GROUP_DEF_FILE)
    parser.add_argument("--do_model", action="store_true", default=False)
    parser.add_argument("--use_only_trials_with_update", action="store_true", default=False)
    parser.add_argument("--do_control", action="store_true", default=False)
    args = parser.parse_args()
    run_with_args(args.group_def_file,
        use_only_trials_with_update=args.use_only_trials_with_update,
        do_model=args.do_model,
        do_control=args.do_control)