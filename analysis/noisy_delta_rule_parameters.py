"""
Compute parameter values of the noisy delta rule model at the subject-level
using analytical formulae, and plot the distribution of parameters across subjects.

This is used to determine sensible parameter values to use for simulations
of the noisy delta rule model in subsequent analyses.
"""

import data_analysis_utils as dana
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plot_utils as plut

FNAME_PREFIX = "noisy_delta_rule_parameters"

def compute_params_per_sub(sub_sessions):
    params_per_sub = []
    for i_sub, sessions in enumerate(sub_sessions):
        arrays = dana.get_data_arrays_from_sessions(sessions,
            keys=["learning_rate", "error", "update"])
        lrs = arrays["learning_rate"]
        filt = ~np.isnan(lrs)
        lrs = lrs[filt]
        errors = arrays["error"][filt]
        updates = arrays["update"][filt]
        params = {}
        # Compute learning rate parameter for this subject as their mean apparent learning rate.
        # This is mathematically justified because:
        # E[apparent learning rate] of a noisy delta rule model
        #   = its learning rate parameter
        # where E is the expectation.
        params["lr"] = np.mean(lrs)
        # For the version of the model with a constant noise level:
        # Compute the noise level parameter for this subject as the standard deviation
        # of the difference between their actual update and the update of the noise-free delta rule.
        # This is mathematically justified because
        # SD[update - learning rate parameter * error] of a noisy delta rule model with constant noise level
        #  = its noise level parameter
        params["noise_level_constant"] = np.std(updates - params["lr"] * errors)
        # For the version of the model with a noise level proportional to the error:
        # Compute the noise level parameter as for the constant noise level version,
        # but dividing the difference between the actual and the noise-free update by
        # magnitude of the error. This is mathematically justified for the same reason.
        params["noise_level_prop-error"] = np.std(
            (updates - params["lr"] * errors) / np.abs(errors))
        params_per_sub += [params]
    return params_per_sub

def run_with_args(group_def_file=dana.ADA_LEARN_GROUP_DEF_FILE):
    data_files = dana.get_subject_data_files_from_group_def(group_def_file)
    output_dir = dana.get_output_dir_for_group_def_file(group_def_file,
        create_if_needed=True)
    task_to_sub_sessions = dana.get_task_to_subject_sessions(data_files)

    for task, sub_sessions in task_to_sub_sessions.items():
        params_per_sub = compute_params_per_sub(sub_sessions)

        # Extract values into separate lists
        learning_rates = [params["lr"] for params in params_per_sub]
        constant_noise_levels = [params["noise_level_constant"] for params in params_per_sub]
        prop_error_noise_levels = [params["noise_level_prop-error"] for params in params_per_sub]

        plut.setup_mpl_style()

        for noise_level_type in ["constant", "prop-error"]:
            xlabel = r"$\eta$"
            noise_level_params = (constant_noise_levels if noise_level_type == "constant"
                else prop_error_noise_levels)
            ylabel = (r"$\sigma_{\epsilon}$" if noise_level_type == "constant"
                else r"$\zeta$")

            figsize = (plut.A4_PAPER_CONTENT_WIDTH / 4, plut.A4_PAPER_CONTENT_WIDTH / 4)
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
            ax.plot(learning_rates, noise_level_params, 'o', ms=1.5,
                color=plut.NOISY_DR_COLORS[noise_level_type],
                alpha=0.8)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))

            figname = f"{FNAME_PREFIX}_noise-level-{noise_level_type}_{task}"
            for ext in ['png', 'pdf']:
                figpath = dana.get_path(output_dir, figname, ext)
                plut.save_figure(fig, figpath)

if __name__ == '__main__':
    run_with_args()