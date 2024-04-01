"""
Make a summary table to check data sanity and quality for an individual subject.
"""

import argparse
import data_analysis_utils as dana
import json
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import plot_utils as plut

SUMMARY_TABLE_WIDTH = 3.5
SUMMARY_TABLE_ROW_HEIGHT = 0.3
SUMMARY_TABLE_FONTSIZE = 10

def str_from_ms(ms, include_s=True, include_ms=False):
    ms = int(ms)
    one_s_ms = 1000
    one_min_ms = (60 * one_s_ms)
    t_min = ms // one_min_ms
    t_rem_s = (ms % one_min_ms) // one_s_ms
    t_rem_ms = ms % one_s_ms
    components = []
    if t_min > 0:
        components += [f"{t_min:d}min"]
    if t_rem_s > 0 and include_s:
        components += [f"{t_rem_s:d}s"]
    if t_rem_ms > 0 and include_ms:
        components += [f"{t_rem_ms:d}ms"]
    return " ".join(components)

def run_with_args(data_file):
    # create output directory if needed for given data file
    output_dir = dana.get_output_dir_for_data_file(data_file,
        create_if_needed=True)

    # load the data
    data = dana.load_data(data_file)
    sessions_per_task = dana.get_sessions_per_task(data)
    n_tasks = len(sessions_per_task)
    subject_id = sessions_per_task[0][0].get("subjectId")
    data_per_task = {
        k: [] for k in ["task_name",
            "n_sessions_started",
            "n_sessions_completed",
            "mean_score_frac",
            "mean_update_freq",
            "mean_lr_above_clean_bounds_freq",
            "mean_lr_below_clean_bounds_freq"]
    }
    for i_task, sessions in enumerate(sessions_per_task):
        task_name = sessions[0].get("taskName")
        n_trials = sessions[0].get("nTrials")
        n_sessions = len(sessions)
        n_sessions_started = np.sum([session["startTime"] > 0 for session in sessions])
        n_sessions_completed = np.sum([session["endTime"] > 0 for session in sessions])
        score_fracs = np.empty(n_sessions_completed)
        estimates = np.empty((n_sessions_completed, n_trials))
        outcomes = np.empty((n_sessions_completed, n_trials))
        for i in range(n_sessions_completed):
            score_fracs[i] = sessions[i]["scoreFrac"]
            estimates[i, :] = sessions[i]["estimate"]
            outcomes[i, :] = sessions[i]["outcome"][0:n_trials]

        estimates_with_prior = np.empty((n_sessions_completed, n_trials + 1))
        estimates_with_prior[:, 0] = 0.5
        estimates_with_prior[:, 1:] = estimates

        measures = dana.estimate_update_measures(estimates, outcomes)
        lrs_raw = measures["learning_rates_raw"]
        was_estimate_updated = measures["was_estimate_updated"]
        is_lr_above_clean_bounds = (lrs_raw > dana.LR_CLEAN_MAX)
        is_lr_below_clean_bounds = (lrs_raw < dana.LR_CLEAN_MIN)

        mean_update_freq_per_sess = np.mean(was_estimate_updated, axis=1)
        mean_score_frac = score_fracs.mean()
        mean_update_freq = mean_update_freq_per_sess.mean()
        mean_lr_above_clean_bounds_freq = np.mean(is_lr_above_clean_bounds)
        mean_lr_below_clean_bounds_freq = np.mean(is_lr_below_clean_bounds)
        
        data_per_task['task_name'] += [task_name]
        data_per_task['n_sessions_started'] += [n_sessions_started]
        data_per_task['n_sessions_completed'] += [n_sessions_completed]
        data_per_task['mean_score_frac'] += [mean_score_frac]
        data_per_task['mean_update_freq'] += [mean_update_freq]
        data_per_task['mean_lr_above_clean_bounds_freq'] += [mean_lr_above_clean_bounds_freq]
        data_per_task['mean_lr_below_clean_bounds_freq'] += [mean_lr_below_clean_bounds_freq]
    
    study_run_events = data["studyRunEvents"]

    ttc_consent_ms = (study_run_events["acceptConsent"][0][0]
        - study_run_events["arriveStudyHomepage"][0][0])
    data_per_task['ttc_instructions_ms'] = []
    data_per_task['ttc_between_task_ms'] = []
    for i_task in range(n_tasks):
        if i_task == 0:
            start = study_run_events["acceptConsent"][0][0]
        else:
            for ev in study_run_events["showTaskInstructionIndex"]:
                inst_idx, task_idx = ev[1]
                if (task_idx == i_task) and (inst_idx == 0):
                    start = ev[0]
                    break
            data_per_task['ttc_between_task_ms'] += [start
                - sessions_per_task[i_task-1][-1]["endTime"]]
        end = sessions_per_task[i_task][0]["startTime"]
        data_per_task['ttc_instructions_ms'] += [end - start]
    data_per_task['ttc_task_ms'] = []
    for i_task, sessions in enumerate(sessions_per_task):
        n_sessions_completed = data_per_task['n_sessions_completed'][i_task]
        n_sessions_started = data_per_task['n_sessions_started'][i_task]
        if (n_sessions_completed == n_sessions_started):
            ttc_task_ms = (sessions[n_sessions_started - 1]["endTime"]
                - sessions[0]["startTime"])
        else:
            ttc_task_ms = (sessions[n_sessions_started - 1]["events"]["trialStart"][-1][0]
                - sessions[0]["startTime"])
        data_per_task['ttc_task_ms'] += [ttc_task_ms]

    if "startStudyDataUpload" in study_run_events:
        total_time_until_upload_ms = (study_run_events["startStudyDataUpload"][0][0]
            - study_run_events["arriveStudyHomepage"][0][0])
    elif "startIncompleteStudyDataUpload" in study_run_events:
        total_time_until_upload_ms = (study_run_events["startIncompleteStudyDataUpload"][0][0]
            - study_run_events["arriveStudyHomepage"][0][0])
    else:
        total_time_until_upload_ms = -1

    totalMoneyBonus = data["totalMoneyBonus"]

    plut.setup_mpl_style()

    summary_table = [
        [f"Subject id: {subject_id}"],
    ]
    for i in range(n_tasks):
        if n_tasks > 1:
            summary_table += [
                [f"Task {i+1}: {data_per_task['task_name'][i]}"],
            ]
        summary_table += [
            [f"Sessions completed: {data_per_task['n_sessions_completed'][i]}"],
            [f"Score (avg): {data_per_task['mean_score_frac'][i]*100:.0f}%"],
            [f"Update frequency: {data_per_task['mean_update_freq'][i]*100:.0f}% of trials"],
            [f"Learning rate < {dana.LR_CLEAN_MIN}: {data_per_task['mean_lr_below_clean_bounds_freq'][i]*100:.0f}% of trials"],
            [f"Learning rate > {dana.LR_CLEAN_MAX}: {data_per_task['mean_lr_above_clean_bounds_freq'][i]*100:.0f}% of trials"],
        ]
    summary_table += [[f"Time to complete consent: {str_from_ms(ttc_consent_ms)}"],]
    for i in range(n_tasks):
        task_str = f"task {i+1}" if n_tasks > 1 else "task"
        if i > 0:
            summary_table += [
                [f"Time between task {i} and {i+1}: {str_from_ms(data_per_task['ttc_between_task_ms'][i-1])}"],
            ]
        summary_table += [
            [f"Time to complete {task_str} instructions: {str_from_ms(data_per_task['ttc_instructions_ms'][i])}"],
            [f"Time to complete {task_str} sessions: {str_from_ms(data_per_task['ttc_task_ms'][i])}"],
        ]
    summary_table += [
        [f"Total time until upload: {str_from_ms(total_time_until_upload_ms)}"],
    ]
    summary_table += [[f"Total money bonus: £{totalMoneyBonus:.2f}"]]

    table_figsize = (SUMMARY_TABLE_WIDTH, SUMMARY_TABLE_ROW_HEIGHT * len(summary_table))
    fig = plt.figure(figsize=table_figsize, constrained_layout=True)
    ax = fig.gca()
    table = ax.table(summary_table, bbox=[0,0,1,1],
        cellLoc='left',
        edges='open')
    table.auto_set_font_size(False)
    table.set_fontsize(SUMMARY_TABLE_FONTSIZE)
    ax.axis('off') # hide axes
    figpath = op.join(output_dir, "summary_table.pdf")
    fig.savefig(figpath)
    print(f"Figure saved at {figpath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file")
    args = parser.parse_args()
    data_file = args.data_file
    run_with_args(data_file)
