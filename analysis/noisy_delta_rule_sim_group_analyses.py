"""
Simulate the noisy delta rule model and perform the same analyses of the apparent
learning rate as for subjects (analysis around change points, and modulation
by the normative model's cpp and uncertainty).
"""

import data_analysis_utils as dana
import itertools
import learning_rate_around_cp_group
import learning_rate_around_cp_subject
import learning_rate_regression_cpp_uncertainty_group
import noisy_delta_rule_parameters
import matplotlib.pyplot as plt
import model_learner as model
import numpy as np
import plot_utils as plut
import scipy.stats

FNAME_PREFIX = "noisy_delta_rule_sim"

# These are roughly the ylims that are obtained for the subjects' plots
lr_around_cp_ylims = {"ada-prob": (0.055, 0.117), "ada-pos": (0.225, 0.78)}
reg_weights_ylims = {"ada-prob": (None, 0.163), "ada-pos": (None, 0.465)}

def run_with_args(group_def_file=dana.ADA_LEARN_GROUP_DEF_FILE):
    data_files = dana.get_subject_data_files_from_group_def(group_def_file)
    output_dir = dana.get_output_dir_for_group_def_file(group_def_file,
        create_if_needed=True)
    task_to_sub_sessions = dana.get_task_to_subject_sessions(data_files)
    np.random.seed(0)
    for task, sub_sessions in task_to_sub_sessions.items():
        # Compute parameter values to use for simulations (one set of parameter per subject)
        params_per_sub = noisy_delta_rule_parameters.compute_params_per_sub(sub_sessions)

        # Simulate the group analyses with the obtained parameters
        noise_level_types = ["constant", "prop-error"]
        sim_results = {k: {} for k in noise_level_types}
        fname_prefix = f"{task}_"
        for noise_level_type in noise_level_types:
            # These are the functions used to compute the estimates of
            # the noisy delta-rule model with the set of parameter obtained
            # for a given subject
            estimate_funs = [(lambda params, noise_level_type: 
                # the above lambda on (params, noise_level_type)
                # is used so that these variables are evaluated
                # immediately when the below lambda is created,
                # rather than only later when the lambda is called
                lambda outcomes: model.noisy_delta_rule_estimates_from_outcomes(
                    outcomes, params["lr"],
                    noise_level_type=noise_level_type,
                    noise_level_param=params[f"noise_level_{noise_level_type}"],
                    do_clip_to_bounds=False))(params, noise_level_type)
                    for params in params_per_sub]

            # Simulate the learning_rate_around_cp analysis
            window_around_cp = dana.get_window_around_cp_for_sessions(sub_sessions[0])
            noisydr_lr_avg_around_cp_per_sub = \
                learning_rate_around_cp_group.compute_lr_avg_around_cp_per_sub(
                    sub_sessions, window_around_cp,
                    estimate_funs=estimate_funs)
            noisydr_lr_avg_around_cp_groupmean = np.nanmean(
                noisydr_lr_avg_around_cp_per_sub, axis=0)
            noisydr_lr_avg_around_cp_groupsem = scipy.stats.sem(
                noisydr_lr_avg_around_cp_per_sub, axis=0, nan_policy="omit")
            sim_results[noise_level_type]["lr_around_cp"] = {
                "groupmean": noisydr_lr_avg_around_cp_groupmean,
                "groupsem": noisydr_lr_avg_around_cp_groupsem,
                }

            plut.setup_mpl_style()
            color = plut.NOISY_DR_COLORS[noise_level_type]

            # Simulate the learning_rate_regression_cpp_uncertainty analysis
            reg_weights_per_sub = learning_rate_regression_cpp_uncertainty_group.compute_reg_weights_per_sub(
                sub_sessions, estimate_funs=estimate_funs)
            reg_weights_stats = learning_rate_regression_cpp_uncertainty_group.compute_stats_on_reg_weights(
                reg_weights_per_sub)
            sim_results[noise_level_type]["reg_weights"] = {
                "stats": reg_weights_stats}
            # Plot the results of the learning_rate_regression_cpp_uncertainty analysis
            figsize = (plut.A4_PAPER_CONTENT_WIDTH / 4,
                       plut.A4_PAPER_CONTENT_WIDTH / 4)
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
            learning_rate_regression_cpp_uncertainty_group.plot_on_ax(ax, reg_weights_stats,
                small_label=True,
                show_sig=False,
                color=color,
                )
            ax.set_ylim(reg_weights_ylims[task])
            if ax.get_ylim()[0] < 0:
                ax.axhline(y=0, ls="-", color=plut.BLACK_COLOR)
            figname = f"{FNAME_PREFIX}_learning-rate_regression_weights_noise-level-{noise_level_type}_{task}"
            for ext in ['png', 'pdf']:
                figpath = dana.get_path(output_dir, figname, ext)
                plut.save_figure(fig, figpath, do_fit_axis_labels=True)

        # Plot the results of the learning_rate_around_cp for both types of noise on the same figure
        window_around_cp = dana.get_window_around_cp_for_sessions(sub_sessions[0])
        fig = learning_rate_around_cp_subject.make_figure(window_around_cp,
                sim_results["constant"]["lr_around_cp"]["groupmean"],
                sim_results["constant"]["lr_around_cp"]["groupsem"],
                task=task, color=plut.NOISY_DR_COLORS["constant"], label=plut.NOISY_DR_TYPE_LABEL["constant"])
        ax = fig.gca()
        learning_rate_around_cp_subject.plot_lr_on_ax(ax, window_around_cp,
            sim_results["prop-error"]["lr_around_cp"]["groupmean"],
            sim_results["prop-error"]["lr_around_cp"]["groupsem"],
            color=plut.NOISY_DR_COLORS["prop-error"], label=plut.NOISY_DR_TYPE_LABEL["prop-error"])
        ax.set_ylim(lr_around_cp_ylims[task])
        leg = ax.legend(title="Noise level")
        leg._legend_box.align = "left"
        figname = f"{FNAME_PREFIX}_learning-rate_around_cp_{task}"
        for ext in ['png', 'pdf']:
            figpath = dana.get_path(output_dir, figname, ext)
            plut.save_figure(fig, figpath)
                

if __name__ == '__main__':
    run_with_args()
