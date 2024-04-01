"""
Script to build an aggregated dataset for a study, where all subjects
are described within the same file, from the original data, which is made of one
data file per subject. The dataset is structured such that data that are
uniquely defined at the level of a trial, a session, or a subject are contained
in their respective table.

Input:
- A 'group_def' csv file containing the list of the individual subject data,
as they collected during the task and saved on the server.
- The path of the directory to create in which to put the dataset.

Ouput: A new directory at the given path containing four csv table files:
- one table for data defined at the "outcome" level
    (i.e. the observation level, or trial level) 
    e.g. the value of the observation at this time step of the sequence,
    and the value of the subject's estimate after receiving that observation.
- one table for data defined at the session level.
    e.g. the id of the sequence that was presented to the subject.
- one table for data defined at the subject level.
    e.g. the subject's id and the time they took to complete the experiment.
- one table describing the experiment parameters that were used for each task.
    e.g. the number of sessions that subjects performed for each task.
"""

import argparse
import data_analysis_utils as dana
import datetime
import dateutil.parser
import numpy as np
import os.path as op
import pandas as pd

FNAME_DATA_SUBJECT_LEVEL = "data_subject-level.csv"
FNAME_DATA_SESSION_LEVEL = "data_session-level.csv"
FNAME_DATA_OUTCOME_LEVEL = "data_outcome-level.csv"
FNAME_TASK_PARAMETERS = "task-parameters.csv"

def run_with_args(group_def_file, output_dir_path):
    dana.create_dir_if_needed(output_dir_path)
    subject_data_files = dana.get_subject_data_files_from_group_def(group_def_file)
    create_subject_level_data_file(subject_data_files, output_dir_path)
    create_session_level_data_file(subject_data_files, output_dir_path)
    create_outcome_level_data_file(subject_data_files, output_dir_path)
    create_task_parameters_file(subject_data_files[0], output_dir_path)

def create_subject_level_data_file(subject_data_files, output_dir_path):
    rows = []
    for data_file in subject_data_files:
        row = {}
        data = dana.load_data(data_file)
        sessions_per_task = dana.get_sessions_per_task(data)
        fst_sess = sessions_per_task[0][0]

        row["subject"] = fst_sess["subjectId"]
        
        study_run_events = data["studyRunEvents"]
        t_start_exp_ms = study_run_events["acceptConsent"][0][0]
        t_upload_ms = study_run_events["startStudyDataUpload"][0][0]
        
        date_fst_sess = dateutil.parser.isoparse(fst_sess["dateCreated"])
        t_fst_sess_from_start_exp_ms = (fst_sess["startTime"] -  t_start_exp_ms)
        date_start_exp = (date_fst_sess
            - datetime.timedelta(milliseconds=t_fst_sess_from_start_exp_ms))
        row["date"] = date_start_exp.isoformat()
        
        row["completion_time_min"] = (t_upload_ms - t_start_exp_ms) / (1e3 * 60)

        studyName = data.get("studyName", None)
        if studyName == "ada-pos-prob":
            row["task_order"] = "pos-prob"
        elif studyName == "ada-prob-pos":
            row["task_order"] = "prob-pos"

        rows += [row]

    out_path = op.join(output_dir_path, FNAME_DATA_SUBJECT_LEVEL)
    save_file_rows(rows, out_path, name="subject-level data")

def create_session_level_data_file(subject_data_files, output_dir_path):
    rows = []
    for data_file in subject_data_files:
        data = dana.load_data(data_file)
        sessions_per_task = dana.get_sessions_per_task(data)
        for sessions in sessions_per_task:
            subject = sessions[0]["subjectId"]
            key_by_col= {
                "task": "taskName",
                "score": "scoreFrac",
                "sequence_id": "sequenceIdx"
            }
            arrays = dana.get_data_arrays_from_sessions(sessions,
                keys=list(key_by_col.values()))
            for i_sess in range(len(sessions)):
                row = {"subject": subject}
                row["task"] = arrays[key_by_col["task"]]
                row["session_idx"] = i_sess
                for col in ["score", "sequence_id"]:
                    row[col] = arrays[key_by_col[col]][i_sess]
                rows += [row]

    out_path = op.join(output_dir_path, FNAME_DATA_SESSION_LEVEL)
    save_file_rows(rows, out_path, name="session-level data")

def create_outcome_level_data_file(subject_data_files, output_dir_path):
    rows = []
    for data_file in subject_data_files:
        data = dana.load_data(data_file)
        sessions_per_task = dana.get_sessions_per_task(data)
        for sessions in sessions_per_task:
            subject = sessions[0]["subjectId"]
            task = sessions[0]["taskName"]
            n_outcomes = sessions[0]["nTrials"]
            outcome_key_by_col= {
                "outcome": "outcome",
                "estimate": "estimate",
                "hidden_parameter": ("p1" if task == "ada-prob"
                    else "mean"),
                "did_change_point_occur": ("didP1Change" if task == "ada-prob"
                    else "didMeanChange"),
            }
            arrays = dana.get_data_arrays_from_sessions(sessions,
                keys=(list(outcome_key_by_col.values()) + ["sequenceIdx"]))
            for i_sess in range(len(sessions)):
                for i_outcome in range(n_outcomes):
                    row = {
                        "subject": subject,
                        "task": task,
                        "session_idx": i_sess,
                        "outcome_idx": i_outcome
                    }
                    for col, key in outcome_key_by_col.items():
                        row[col] = arrays[key][i_sess, i_outcome]
                    row["sequence_id"] = arrays["sequenceIdx"][i_sess]
                    rows += [row]

    out_path = op.join(output_dir_path, FNAME_DATA_OUTCOME_LEVEL)
    save_file_rows(rows, out_path, name="outcome-level data")

def create_task_parameters_file(subject_data_file, output_dir_path):
    data = dana.load_data(subject_data_file)
    n_tasks = data.get("nTasks", 1)
    if n_tasks == 1:
        sessions = data["taskSessionDataArray"]
        sessions_by_task = {sessions[0]["taskName"]: sessions}
    else:
        sessions_by_task = {}
        for sessions in data["taskSessionDataArrays"]:
            sessions_by_task[sessions[0]["taskName"]] = sessions
    rows = []
    for task, sessions in sessions_by_task.items():
        params = {}
        params["task"] = task
        params["n_sessions"] = len(sessions)
        params["n_outcomes"] = sessions[0]["nTrials"]
        params["p_c"] = sessions[0]["sequencePC"]
        if task == "ada-pos":
            params["outcome_sd"] = sessions[0]["sequenceStd"]
        else:
            params["outcome_sd"] = np.nan
        rows += [params]
    
    out_path = op.join(output_dir_path, FNAME_TASK_PARAMETERS)
    save_file_rows(rows, out_path, name="task parameters")

def save_file_rows(rows, out_path, name=""):
    df = pd.DataFrame(rows)
    df.to_csv(out_path)
    print(f"Created {name} file at: {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group_def_file")
    parser.add_argument("output_dir_path")
    args = parser.parse_args()
    run_with_args(args.group_def_file,
        args.output_dir_path)
