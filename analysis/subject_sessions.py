"""
Plot the individual sequences (outcomes, estimates, and generative value)
for each session of the given subject data file.
"""

import argparse
import data_analysis_utils as dana
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import plot_utils as plut

FIGSIZE = (plut.A4_PAPER_CONTENT_WIDTH * 1/2, plut.DEFAULT_HEIGHT)
FIGSIZE_VERTICAL = (plut.A4_PAPER_CONTENT_WIDTH * 1/2, 75 * 0.3)
XAXIS_HEIGHT = 120 / 300

TIME_BETWEEN_SESSION_START_AND_FIRST_TRIAL_MS = 1000

def run_with_args(data_file,
    with_generative=False,
    with_slider_movements=False,
    i_sessions=None,
    exts=["png"]):
    vertical_layouts = [False, True] if with_slider_movements else [False]
    for use_vertical_layout in vertical_layouts:
        plot_figures(data_file,
            with_generative=with_generative,
            with_slider_movements=with_slider_movements,
            use_vertical_layout=use_vertical_layout,
            i_sessions=i_sessions,
            exts=exts)

def plot_figures(data_file,
    with_generative=False,
    with_slider_movements=False,
    use_vertical_layout=False,
    i_sessions=None,
    exts=["png"]):
    is_wip = with_slider_movements or use_vertical_layout
    output_dir = dana.get_output_dir_for_data_file(data_file,
            create_if_needed=True, is_wip=is_wip)

    figsize = FIGSIZE_VERTICAL if use_vertical_layout else FIGSIZE
    if use_vertical_layout:
        figsize = (figsize[0] + plut.TWIN_AXIS_PAD + XAXIS_HEIGHT), figsize[1]
    else:
        figsize = figsize[0], (figsize[1] + plut.TWIN_AXIS_PAD + XAXIS_HEIGHT)
    plut.setup_mpl_style()

    data = dana.load_data(data_file)
    sessions_per_task = dana.get_sessions_per_task(data)
    n_tasks = len(sessions_per_task)
    subject_id = sessions_per_task[0][0].get("subjectId")

    if with_slider_movements:
        sessions_event_segments = dana.load_sessions_event_segments_with_data_file(data_file)
        figname_prefix = f"sequence-with-slider-movements_sub-{subject_id}"
    else:
        figname_prefix = f"sequence_sub-{subject_id}"
    if with_generative:
        figname_prefix += "_with-generative"
    if use_vertical_layout:
        figname_prefix += "_vertical-layout"

    for i_task, sessions in enumerate(sessions_per_task):
        task_name = sessions[0]["taskName"]
        gen_key = dana.get_gen_key_for_task(task_name)
        if task_name == "ada-prob":
            change_key = 'didP1Change'
        elif task_name == "ada-pos":
            change_key = 'didMeanChange'
        arrays = dana.get_data_arrays_from_sessions(sessions, keys=[
            "estimate", "outcome", gen_key, change_key])
        n_sessions, n_trials = arrays["outcome"].shape
        trials = np.arange(1, n_trials + 1)
        trials_with_prior = np.array([0] + list(trials))
        lim_trials = trials[0] - 1, trials[-1] + 1
        label_trials = "Observations"
        estimate_trial_offset = (0.99 if with_slider_movements else 0.5)
        lim_estimates = (-0.01, 1.01)
        if task_name == "ada-prob":
            label_estimates = "Probability p(blue)"
            alpha_estimate = 0.8
        elif task_name == "ada-pos":
            label_estimates = "Position"
            alpha_estimate = None
        if i_sessions is None:
            i_sessions = range(n_sessions)
        for i_sess in i_sessions:
            outcomes = arrays["outcome"][i_sess]
            estimates_with_prior = np.array([dana.ESTIMATE_PRIOR] + list(arrays["estimate"][i_sess]))
            generatives = arrays[gen_key][i_sess]
            change_indices = np.argwhere(arrays[change_key][i_sess])[1:].squeeze()
            plot_args_s = []
            # Add outcomes in elements to plot
            if task_name == "ada-prob":
                for b in [0, 1]:
                    color = (plut.ADA_PROB_OUTCOME_1_COLOR if b == 1
                        else plut.ADA_PROB_OUTCOME_0_COLOR)
                    filt = (outcomes == b)
                    plot_args_s += [dict(x=trials[filt], y=([0.5] * filt.sum()),
                        fmt='.', ms=2, color=color)]
            elif task_name == "ada-pos":
                plot_args_s += [dict(x=trials, y=outcomes,
                        fmt='.', ms=1., color=plut.ADA_POS_OUTCOME_COLOR)]
            if with_generative:
                # Add generative value
                # add one value before each change point to plot a rectangular step
                trials_with_extra = np.insert(trials, change_indices, trials[change_indices])
                generatives_with_extra = np.insert(generatives, change_indices, generatives[change_indices-1])
                plot_args_s += [dict(x=trials_with_extra, y=generatives_with_extra, fmt=':',
                    lw=0.5, color=plut.GENERATIVE_COLOR)]
            # Add subjects' estimates
            plot_args_s += [dict(x=trials_with_prior+estimate_trial_offset,
                y=estimates_with_prior, fmt='-', color=plut.SUBJECT_COLOR, alpha=alpha_estimate,
                label=plut.SUBJECT_LABEL)]
            
            if with_slider_movements:
                # Add line segments to show slider movements
                event_segments = sessions_event_segments[i_sess]
                assert event_segments[0]["event_type"] == "session"
                session_start = event_segments[0]["start_time"]
                current_trial_segment = {
                    "start_time": session_start,
                    "end_time": TIME_BETWEEN_SESSION_START_AND_FIRST_TRIAL_MS,
                    "index": -1
                }
                last_trajectory_trial_frac = 0
                last_trajectory_estimate = dana.ESTIMATE_PRIOR
                for segment in event_segments:
                    if (segment["event_type"] == "trial"):
                        current_trial_segment = segment
                    if (segment["event_type"] == "slider_movement"):
                        trajectory_times = [t for t, e in segment["estimate_trajectory"]]
                        trajectory_estimates = [e for t, e in segment["estimate_trajectory"]]
                        trajectory_times = np.array(trajectory_times)
                        trajectory_times_frac_within_trial = (
                            (trajectory_times - current_trial_segment["start_time"])
                            / (current_trial_segment["end_time"] - current_trial_segment["start_time"]))
                        trajectory_trial_fracs = (trajectory_times_frac_within_trial
                            + current_trial_segment["index"] + 1)
                        trajectory_color = plut.COLORS_NEUTRAL_GRAY[5]
                        # Dotted line to connect different movement segments
                        # (no movement was made in between)
                        plot_args_s += [dict(x=[last_trajectory_trial_frac, trajectory_trial_fracs[0]],
                            y=[last_trajectory_estimate, trajectory_estimates[0]],
                            fmt=':', lw=0.5, color=trajectory_color)]
                        # Continuous line to show one movement segment
                        plot_args_s += [dict(x=trajectory_trial_fracs, y=trajectory_estimates,
                            fmt='-', lw=0.5, color=trajectory_color, alpha=alpha_estimate)]

                        last_trajectory_estimate = trajectory_estimates[-1]
                        last_trajectory_trial_frac = trajectory_trial_fracs[-1]

            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
            xlim = lim_estimates if use_vertical_layout else lim_trials
            ylim = lim_trials if use_vertical_layout else lim_estimates
            xlabel = label_estimates if use_vertical_layout else label_trials
            ylabel = label_trials if use_vertical_layout else label_estimates
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            # Display change points
            for i in change_indices:
                if use_vertical_layout:
                    ax.axhline(y=trials[i], ls=':', lw=1., color=plut.GENERATIVE_COLOR)
                else:
                    ax.axvline(x=trials[i], ls=':', lw=1., color=plut.GENERATIVE_COLOR)
            if use_vertical_layout: # flip x and y values before plotting
                for args in plot_args_s:
                    y = args["x"]
                    args["x"] = args["y"]
                    args["y"] = y
            for args in plot_args_s:
                x = args.pop("x")
                y = args.pop("y")
                fmt = args.pop("fmt")
                ax.plot(x, y, fmt, **args)
            # Add twin axis to show time and set common tick values for outcomes and time
            tick_step = 10 # 10 outcomes = 15 s
            tickvals = trials_with_prior[::tick_step]
            ticklabels = [f"{t*dana.SOA_S:.0f}" for t in tickvals]
            twin_axis_offset_pt = (plut.TWIN_AXIS_PAD + XAXIS_HEIGHT) * 72
            if use_vertical_layout:
                ax_time = ax.twiny()
                ax_time.spines["left"].set_position(("outward",
                    twin_axis_offset_pt))
                ax_time.yaxis.set_label_position("left")
                ax_time.yaxis.set_ticks_position("left")
                ax_time.set_ylim(ax.get_ylim())
                ax_time.set_yticks(tickvals, labels=ticklabels)
                ax_time.set_ylabel(plut.SEC_LABEL)
            else:
                ax_time = ax.twiny()
                ax_time.spines["bottom"].set_position(("outward",
                    twin_axis_offset_pt))
                ax_time.xaxis.set_label_position("bottom")
                ax_time.xaxis.set_ticks_position("bottom")
                ax_time.set_xlim(ax.get_xlim())
                ax_time.set_xticks(tickvals, labels=ticklabels)
                ax_time.set_xlabel(plut.SEC_LABEL)
            # ax.legend()
            if use_vertical_layout:
                # Display time from top to bottom
                ax.invert_yaxis()
                # Show x axis on the top instead of the bottom
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(True)
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top') 

            if n_tasks > 1:
                figname = f"{figname_prefix}_task-{i_task+1:d}_isess-{i_sess:d}"
            else:
                figname = f"{figname_prefix}_isess-{i_sess:d}"
            for ext in exts:
                figpath = dana.get_path(output_dir, figname, ext)
                plut.save_figure(fig, figpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file")
    parser.add_argument("--with_generative",
        action='store_true', default=False)
    parser.add_argument("--with_slider_movements",
        action='store_true', default=False)
    parser.add_argument("--i_sessions", nargs='*', type=int, default=None) # None means all sessions
    parser.add_argument("--exts", nargs='+', default=["png"])
    args = parser.parse_args()
    data_file = args.data_file
    run_with_args(data_file,
        with_generative=args.with_generative,
        with_slider_movements=args.with_slider_movements,
        i_sessions=args.i_sessions,
        exts=args.exts
        )
    
