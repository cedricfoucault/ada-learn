"""
Analyze relationship between the subject's learning rate
and model variable (cpp, uncertainty) at the group level.
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
import learning_rate_relationship_cpp_uncertainty_subject
import plot_utils as plut
import scipy.stats

FNAME_PREFIX = "learning-rate_relationship"

FIGSIZE_BARPLOT_VERTICAL = (plut.A4_PAPER_CONTENT_WIDTH * 1/4,
    plut.A4_PAPER_CONTENT_WIDTH * 1/2)

XAXIS_HEIGHT = 110 / 300
XAXIS_TO_LEGEND = 14 / 300
LEGEND_HEIGHT = 138 / 300

def run_with_args(group_def_file,
    do_model=False,
    use_only_trials_with_update=False,
    small=False,
    nbins=learning_rate_relationship_cpp_uncertainty_subject.N_BINS_UNCERTAINTY,
    nbins_2d=learning_rate_relationship_cpp_uncertainty_subject.N_BINS_UNCERTAINTY // 2,
    ):
    """
    Parameters:
    - do_model:
        If True, compute the results predicted by the normative model,
        doing the analysis in exactly the same way as for subjects
        but using the model's learning rate rather than the subject's.
    - use_only_trials_with_update:
        If True, all data points where the subject did not perform an update
        (yielding a learning rate equal to 0) will be discarded from the analysis.
    """
    data_files = dana.get_subject_data_files_from_group_def(group_def_file)
    n_subjects = len(data_files)
    output_dir = dana.get_output_dir_for_group_def_file(group_def_file,
                create_if_needed=True)
    task_to_sub_sessions = dana.get_task_to_subject_sessions(data_files)
    n_tasks = len(task_to_sub_sessions)
    
    for task, sub_sessions in task_to_sub_sessions.items():
        # Compute data for each subject
        subs_data = {}
        subs_data["lrs"] = [None for _ in range(n_subjects)]
        subs_data["uncertainty"] = [None for _ in range(n_subjects)]
        subs_data["cpp"] = [None for _ in range(n_subjects)]
        uncertainty_all = [] # used to compute the bins
        cpp_all = [] # used to compute the bins
        for i_sub, sessions in enumerate(sub_sessions):
            res = learning_rate_relationship_cpp_uncertainty_subject.run_with_sessions(
                sessions,
                do_plot=False,
                use_only_trials_with_update=use_only_trials_with_update,
                do_model=do_model)
            for key in ["lrs", "uncertainty", "cpp",
                ]:
                subs_data[key][i_sub] = res[key]
            uncertainty_all += [res["uncertainty"]]
            cpp_all += [res["cpp"]]
        uncertainty_all = np.concatenate(uncertainty_all)
        cpp_all = np.concatenate(cpp_all)

        if do_model:
            plut.setup_mpl_style(fontsize=7)
        else:
            plut.setup_mpl_style()

        # Plot group average learning rate by bin of uncertainty

        if (not do_model):
            _, bin_edges = pd.qcut(uncertainty_all, nbins, retbins=True)
            subs_data["uncertainty_avg_per_bin"] = np.empty((n_subjects, nbins))
            subs_data["lr_avg_per_bin"] = np.empty((n_subjects, nbins))
            for i_sub in range(n_subjects):
                sub_data_per_bin = learning_rate_relationship_cpp_uncertainty_subject.get_data_per_bin(
                    subs_data["uncertainty"][i_sub],
                    subs_data["lrs"][i_sub],
                    bin_edges=bin_edges)
                subs_data["uncertainty_avg_per_bin"][i_sub, :] = sub_data_per_bin["x_avg"]
                subs_data["lr_avg_per_bin"][i_sub, :] = sub_data_per_bin["y_avg"]
            
            subs_uncert_avg_per_bin = subs_data["uncertainty_avg_per_bin"]
            subs_lr_avg_per_bin = subs_data["lr_avg_per_bin"]
            group_uncert_avg_per_bin = np.mean(subs_uncert_avg_per_bin,
                axis=0)
            group_lr_avg_per_bin = np.mean(subs_lr_avg_per_bin, axis=0)
            group_lr_sem_per_bin = scipy.stats.sem(subs_lr_avg_per_bin, axis=0)
            xlabel = plut.get_uncertainty_label(with_unit=(not small))
            ylabel = "Learning rate"
            if small:
                figsize = (plut.A4_PAPER_CONTENT_WIDTH / 4 - 0.5,
                           plut.A4_PAPER_CONTENT_WIDTH / 4 - 0.5)
            else:
                figsize = learning_rate_relationship_cpp_uncertainty_subject.FIGSIZE
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
            if small:
                ax.set_xlabel(xlabel, fontsize=7)
                ax.set_ylabel(ylabel, fontsize=7)
                ax.tick_params(axis='both', which='major', labelsize=7)
                ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            else:
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
            for i_bin in range(len(group_uncert_avg_per_bin)):
                ax.errorbar(group_uncert_avg_per_bin[i_bin],
                    group_lr_avg_per_bin[i_bin],
                    group_lr_sem_per_bin[i_bin],
                    fmt='o', ms=2,
                    color=plut.SUBJECT_COLOR,
                    ecolor=plut.GRAY_COLOR,
                    )
            if use_only_trials_with_update:
                ax.text(0.025, 1.0, plut.ONLY_TRIALS_WITH_UPDATE_LABEL,
                    ha="left", va="top",
                    transform=ax.transAxes)

            figname = FNAME_PREFIX + "_uncertainty"
            if small:
                figname += "_small"
            if use_only_trials_with_update:
                figname += "_use-only-trials-with-update"
            if n_tasks > 1:
                figname += f"_{task}"
            for ext in ["png", "pdf"]:
                figpath = dana.get_path(output_dir, figname, ext)
                plut.save_figure(fig, figpath, do_fit_axis_labels=True)

        # Plot group average learning rate by bin of uncertainty
        # split into lower and higher cpp

        nbins_uncert_2d = nbins_2d
        if do_model:
            NBINS_2D_MODEL_BY_TASK = {"ada-prob": 15, "ada-pos": 4}
            nbins_uncert_2d = NBINS_2D_MODEL_BY_TASK[task]
        _, bin_edges_uncert = pd.qcut(uncertainty_all, nbins_uncert_2d, retbins=True)
        if task == "ada-pos":
            bin_edges_cpp = np.linspace(cpp_all.min(), cpp_all.max(), num=3)
        else:
            _, bin_edges_cpp = pd.qcut(cpp_all, 2, retbins=True)
        for key in (["lr", "uncertainty"]):
            subs_data[key + "_avg_per_bin_2d"] = np.empty((n_subjects, nbins_uncert_2d, 2))
        for i_sub in range(n_subjects):
            lrs = subs_data["lrs"][i_sub]
            uncerts = subs_data["uncertainty"][i_sub]
            cpps = subs_data["cpp"][i_sub]
            for (i_bin_uncert, i_bin_cpp) in itertools.product(range(nbins_uncert_2d), range(2)):
                bin_mask_uncert = dana.get_bin_mask(uncerts, bin_edges_uncert, i_bin_uncert)
                bin_mask_cpp = dana.get_bin_mask(cpps, bin_edges_cpp, i_bin_cpp)
                bin_mask = (bin_mask_uncert & bin_mask_cpp)
                for (key, vals) in [("lr", lrs), ("uncertainty", uncerts)]:
                    subs_data[key + "_avg_per_bin_2d"][i_sub, i_bin_uncert, i_bin_cpp] = np.mean(vals[bin_mask])

        subs_uncert_avg_per_bin_2d = subs_data["uncertainty_avg_per_bin_2d"]
        subs_lr_avg_per_bin_2d = subs_data["lr_avg_per_bin_2d"]
        group_uncert_avg_per_bin_2d = np.mean(subs_uncert_avg_per_bin_2d,
            axis=0)
        group_lr_avg_per_bin_2d = np.mean(subs_lr_avg_per_bin_2d, axis=0)
        if not do_model:
            group_lr_sem_per_bin_2d = scipy.stats.sem(subs_lr_avg_per_bin_2d, axis=0)
        xlabel = plut.get_uncertainty_label()
        ylabel = "Learning rate"
        if do_model:
            figsize = (4.2 / plut.CM_PER_INCH,
                       5.4 / plut.CM_PER_INCH)
        else:
            figsize = (plut.A4_PAPER_CONTENT_WIDTH / 3,
                       (plut.A4_PAPER_CONTENT_WIDTH / 3
                        + LEGEND_HEIGHT + XAXIS_TO_LEGEND))
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if do_model:
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
        levels_cpp = [f"lower ({bin_edges_cpp[0]:.2f} – {bin_edges_cpp[1]:.2f})",
                    f"higher ({bin_edges_cpp[1]:.2f} – {bin_edges_cpp[2]:.2f})"]
        for i_bin_cpp in reversed(range(2)):
            ms = (1+i_bin_cpp)
            color = plut.BLACK_COLOR if do_model else plut.SUBJECT_COLOR
            ax.plot(group_uncert_avg_per_bin_2d[:, i_bin_cpp],
             group_lr_avg_per_bin_2d[:, i_bin_cpp],
             '-o',
             ms=ms, lw=(1+i_bin_cpp-0.5),
             color=color,
             label=levels_cpp[i_bin_cpp])
            if not do_model:
                for i_bin_uncert in range(nbins_uncert_2d):
                    ax.errorbar(group_uncert_avg_per_bin_2d[i_bin_uncert, i_bin_cpp],
                        group_lr_avg_per_bin_2d[i_bin_uncert, i_bin_cpp],
                        group_lr_sem_per_bin_2d[i_bin_uncert, i_bin_cpp],
                        fmt='o', ms=ms,
                        color=color,
                        ecolor=plut.GRAY_COLOR,
                        )
        if do_model:
            figlen_ref = plut.A4_PAPER_CONTENT_WIDTH / 4
            legend_y_frac = -((XAXIS_TO_LEGEND+XAXIS_HEIGHT) /
                ((figlen_ref-XAXIS_HEIGHT)))
            ax.legend(title=plut.CPP_LABEL,
                fontsize=6.5, title_fontsize=7,
                loc='upper left', bbox_to_anchor=(0.0,legend_y_frac),
                borderpad=0.)
        else:
            figlen_ref = plut.A4_PAPER_CONTENT_WIDTH / 3
            legend_y_frac = -((XAXIS_TO_LEGEND+XAXIS_HEIGHT) /
                ((figlen_ref-XAXIS_HEIGHT)))
            ax.legend(title=plut.CPP_LABEL, fontsize=7,
                    loc='upper left', bbox_to_anchor=(0.0,legend_y_frac),
                    borderpad=0.)
        if use_only_trials_with_update:
            ax.text(0.025, 1.0, plut.ONLY_TRIALS_WITH_UPDATE_LABEL, ha="left", va="top",
                transform=ax.transAxes)

        figname = FNAME_PREFIX + "_uncertainty_cpp"
        if use_only_trials_with_update:
            figname += "_use-only-trials-with-update"
        if do_model:
            figname += "_model"
        if n_tasks > 1:
            figname += f"_{task}"
        for ext in ["png", "pdf"]:
            figpath = dana.get_path(output_dir, figname, ext)
            plut.save_figure(fig, figpath, do_fit_axis_labels=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group_def_file", type=str,
        default=dana.ADA_LEARN_GROUP_DEF_FILE)
    parser.add_argument("--do_model", action="store_true", default=False)
    parser.add_argument("--use_only_trials_with_update", action="store_true", default=False)
    args = parser.parse_args()
    run_with_args(args.group_def_file,
        use_only_trials_with_update=args.use_only_trials_with_update,
        do_model=args.do_model)
