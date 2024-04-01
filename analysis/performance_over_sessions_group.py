"""
Analyze subject's performance over the course of the task sessions at the group level.
"""

import argparse
import data_analysis_utils as dana
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import performance_over_sessions_subject
import plot_utils as plut
import scipy.stats

FNAME_PREFIX = "performance_over_sessions"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group_def_file", type=str,
        default=dana.ADA_LEARN_GROUP_DEF_FILE)
    parser.add_argument("--do_plot_individual", action="store_true", default=False)
    args = parser.parse_args()
    group_def_file = args.group_def_file
    do_plot_individual = args.do_plot_individual

    data_files = dana.get_subject_data_files_from_group_def(group_def_file)
    output_dir = dana.get_output_dir_for_group_def_file(group_def_file,
        create_if_needed=True)
    n_subjects = len(data_files)
    task_to_sub_sessions = dana.get_task_to_subject_sessions(data_files)
    n_tasks = len(task_to_sub_sessions)

    plut.setup_mpl_style()

    for task, sub_sessions in task_to_sub_sessions.items():
        subs_score_fracs_per_sess = [[] for _ in range(n_subjects)]
        subs_errors_per_sess = [[] for _ in range(n_subjects)]
        for i_sub in range(n_subjects):
            res = performance_over_sessions_subject.run_with_sessions(sub_sessions[i_sub],
                do_plot=do_plot_individual)
            subs_score_fracs_per_sess[i_sub] = res["score_fracs"]
            subs_errors_per_sess[i_sub] = res["errors"]

        subs_score_fracs_per_sess = np.array(subs_score_fracs_per_sess)
        subs_errors_per_sess = np.array(subs_errors_per_sess)
        error_ub = max(res["error_ub"], subs_errors_per_sess.max())

        metric = "error"
        subs_performances = (subs_score_fracs_per_sess if metric == "score"
            else (-subs_errors_per_sess))

        # Plot performance over sessions
        group_performance_means = np.mean(subs_performances, axis=0)
        group_performances_sems = scipy.stats.sem(subs_performances, axis=0)
        figname = FNAME_PREFIX
        if n_tasks > 1:
            figname += f"_{task}"
        pad_width = 0.08
        figsize = ((plut.A4_PAPER_CONTENT_WIDTH - pad_width) * (1/3 if task == "ada-pos" else 2/3),
                   (plut.A4_PAPER_CONTENT_WIDTH - pad_width) * 1/4
                   )
        SUB_ALPHA = 0.3
        SUB_MS = 1.
        SUB_LW = 0.5
        GROUP_MS = 3
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        for i_sub in range(n_subjects):
            performance_over_sessions_subject.plot_performances_on_axes(ax,
                subs_performances[i_sub], metric=metric, error_ub=error_ub,
                do_setup_axes=False,
                session_color=plut.GRAY_COLOR,
                session_edgecolor=plut.BLACK_COLOR,
                session_alpha=SUB_ALPHA,
                session_ms=SUB_MS, session_lw=SUB_LW,
                show_avg=False
                )
        performance_over_sessions_subject.plot_performances_on_axes(ax,
                group_performance_means, metric=metric, error_ub=error_ub,
                do_setup_axes=True,
                session_fmt='o',
                session_color=plut.GRAY_COLOR, session_edgecolor=plut.BLACK_COLOR,
                session_ms=GROUP_MS, session_lw=1,
                show_avg=False
                )
        ax.tick_params(labelsize=7)
        for ext in ['png', 'pdf']:
            figpath = dana.get_path(output_dir, figname, ext=ext)
            plut.save_figure(fig, figpath)

            
            
