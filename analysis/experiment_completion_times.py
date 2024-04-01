"""
Analyze time taken by participants to complete the experiment
(from the moment the subject gave their informed consent
to the end of the last session of the task).
"""

import argparse
import data_analysis_utils as dana
import numpy as np
import os.path as op
import pandas as pd

OUT_FNAME = "experiment_completion_time_summary.csv"

def run_with_args(group_def_file):
    data_files = dana.get_subject_data_files_from_group_def(group_def_file)
    output_dir = dana.get_output_dir_for_group_def_file(group_def_file,
        create_if_needed=True, is_wip=True)
    n_subjects = len(data_files)

    completion_times_min = np.empty(n_subjects)
    for i_sub, data_file in enumerate(data_files):
        data = dana.load_data(data_file)
        study_run_events = data["studyRunEvents"]
        t_consent_ms = study_run_events["acceptConsent"][0][0]
        t_upload_ms = study_run_events["startStudyDataUpload"][0][0]
        completion_time_min = (t_upload_ms - t_consent_ms) / (1e3 * 60)
        # print(f"completion time: {completion_time_min:.1f} min")
        completion_times_min[i_sub] = completion_time_min
    summary_data = {
        "completion_time_min_median": np.median(completion_times_min),
        "completion_time_min_interquartile": (np.quantile(completion_times_min, 1/4),
            np.quantile(completion_times_min, 3/4)),
        # "completion_time_min_mean": np.mean(completion_times_min),
        # "completion_time_min_std": np.std(completion_times_min),
    }
    print(summary_data)
    out_path = op.join(output_dir, OUT_FNAME)
    pd.Series(summary_data).to_csv(out_path)
    print(f"Summary saved at {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group_def_file", type=str,
        default=dana.ADA_LEARN_GROUP_DEF_FILE)
    args = parser.parse_args()
    run_with_args(args.group_def_file)

