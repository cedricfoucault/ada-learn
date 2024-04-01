"""
Analyze subject's performance over the course of the task sessions at the subject level.
"""

import argparse
import data_analysis_utils as dana
import matplotlib.pyplot as plt
import model_learner as model
import numpy as np
import os.path as op
import plot_utils as plut
import scipy.stats

FIGSIZE = (plut.A4_PAPER_CONTENT_WIDTH / 2., plut.DEFAULT_HEIGHT)

def plot_performances_on_axes(ax, performances, metric="score", error_ub=None,
    show_avg=False, do_setup_axes=True,
    avg_color=plut.GRAY_COLOR, avg_edgecolor=plut.BLACK_COLOR, avg_alpha=1,
    avg_fmt='o', avg_ms=3,
    session_color=plut.GRAY_COLOR, session_edgecolor=plut.GRAY_COLOR, session_alpha=1,
    session_fmt='o-', session_ms=1, session_lw=1):
    n_sessions = performances.shape[0]
    ylabel = "Score" if metric == "score" else "Neg. mean absolute error"
    yvals = performances
    xvals = 1 + np.arange(n_sessions)
    xticklabels = [f"{(i_sess + 1):d}" for i_sess in range(n_sessions)]
    if show_avg:
        yvals = np.append(yvals, np.mean(performances))
        xvals = np.append(xvals, n_sessions + 1)
        xticklabels += ["avg."]
    ylim = (-0.01, 1.01) if metric == "score" else (-error_ub, 0)
    xlabel = "Task session"
    ax.plot(xvals[:n_sessions], yvals[:n_sessions], session_fmt,
        color=session_color, markeredgecolor=session_edgecolor, alpha=session_alpha,
        ms=session_ms, lw=session_lw)
    if show_avg:
        ax.plot(xvals[-1:], yvals[-1:], avg_fmt,
            color=avg_color, markeredgecolor=avg_edgecolor, alpha=avg_alpha,
            ms=avg_ms)
    if do_setup_axes:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(xvals)
        ax.set_xticklabels(xticklabels)
        ax.set_ylim(ylim)
    return ax

def run_with_args(data_file, do_plot=True):
    data = dana.load_data(data_file)
    output_dir = dana.get_output_dir_for_data_file(data_file,
        create_if_needed=True)
    sessions_per_task = dana.get_sessions_per_task(data)
    n_tasks = len(sessions_per_task)
    for i_task, sessions in enumerate(sessions_per_task):
        retval = run_with_sessions(sessions, do_plot=do_plot,
            output_dir=output_dir, i_task=(i_task if n_tasks > 1 else None))
    return retval

def run_with_sessions(sessions, do_plot=True, output_dir=None,
    i_task=None):
    assert dana.all_sessions_completed(sessions)
    n_sessions = len(sessions)
    score_fracs = np.empty(n_sessions)
    errors = np.empty(n_sessions)
    error_ub = dana.get_absolute_error_upper_bound(sessions[0])
    model_errors = np.empty(n_sessions)
    task_name = sessions[0]["taskName"]
    gen_key = dana.get_gen_key_for_task(task_name)
    arrays = dana.get_data_arrays_from_sessions(sessions, keys=[
        "estimate", "outcome", gen_key])
    p_c = dana.get_change_point_probability(sessions[0])
    p1_min, p1_max = dana.get_hidvar_bounds(sessions[0])
    outcomes = arrays["outcome"]
    for i, session in enumerate(sessions):
        errors[i] = dana.get_absolute_errors(session).mean()
        score_fracs[i] = session["scoreFrac"]
        if task_name == "ada-prob":
            model_estimates = model.ideal_learner_prob_estimates_from_outcomes(
                outcomes[i], p_c, estimate_type="mean", p1_min=p1_min, p1_max=p1_max)
        elif task_name == "ada-pos":
            std_gen = dana.get_std_gen(sessions[0])
            model_estimates = model.reduced_learner_pos_estimates_from_outcomes(
                outcomes[i], p_c, std_gen=std_gen)
        model_errors[i] = dana.get_absolute_errors(session,
            custom_estimates=model_estimates).mean()

    # compute performance as the difference between model's and subject's
    # mean absolute error
    performances = model_errors - errors
    retval = {
        "score_fracs": score_fracs,
        "errors": errors,
        "error_ub": error_ub,
        "performances": performances,
         }

    if not do_plot:
        return retval

    plut.setup_mpl_style()

    for metric in ["score", "error"]:
        performances = score_fracs if metric == "score" else (-errors)
        figname = f"performance-{metric}_over_sessions"
        if i_task is not None:
            figname += f"_task-{i_task+1}"
        figname += ".png"
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.gca()
        plot_performances_on_axes(ax, performances, metric=metric, error_ub=error_ub)
        figpath = op.join(output_dir, figname)
        plut.save_figure(fig, figpath)

    return retval

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file")
    args = parser.parse_args()
    data_file = args.data_file
    run_with_args(data_file)
