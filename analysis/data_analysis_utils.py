import json
import model_learner as model
import numpy as np
import os
import os.path as op
import pandas as pd
import pickle

# Make sure that the current working directory is set to the root directory of the repository,
# add it to the search path for modules if necessary, and make sure that the data folder can be found.
root_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not os.path.samefile(os.getcwd(), root_dir_path):
    import sys
    print("Setting the current working directory to the root directory of the repository. "
        "Please run the code from the root of the repository in the future to silence this warning.",
        file=sys.stderr)
    os.chdir(root_dir_path)
if not os.path.exists('data'):
    print("Warning: The 'data' folder is could not be found. "
          "It is expected to be at the root directory of the repository. "
          "Please make sure that the repository is complete.",
          file=sys.stderr)

# This 'group definition' file lists the behavioral data files
# for each of the N=96 subjects that took part in our 'Ada-Learn' study,
# which combines the two tasks (ada-pos, magnitude learning task,
# and ada-prob, probability learning task) in a counterbalanced task order.
# No subjects were excluded.
ADA_LEARN_GROUP_DEF_FILE = "data/group_defs/group_def_ada-learn_23-02-17.csv"

# This file describes the dataset of all the different outcome sequences
# that were presented to subjects in our study
# (100 sequences for ada-pos, 150 sequences for ada-prob).
# For each subject, we randomly sampled from this dataset a subset of sequences
# to present at each session (the sampling was performed without replacement at the subject-level,
# but with replacement at the group level).
# As a result, each outcome sequence was presented multiple times across the subjects.
ADA_LEARN_SEQ_DATA_FILE = "tasks/seq-data/seq-data_pos-prob_ntrials-75_seed-1_nsessions-pos-100-prob-150_std-pos-10by300_pc-pos-1by10-prob-1by20_min-run-length-pos-3-prob-6_min-odd-change-prob-4_apply-min-for-last-run-length-prob_max-run-length-prob-60.csv"

OUTPUT_BASE_DIR = "results"
OUTPUT_GROUP_BASE_DIR = "results"
OUTPUT_WIP_DIR = "wip"


# These threshold values of learning rate were determined a priori
# from the histogram of learning rates obtained in a pilot study of ada-pos.
# We thresholded those learning rates whose density in the histogram was below
# the probability density of a normal distribution at z=2, to avoid aberrant
# learning rate values due to very small denominators (error: estimate - outcome)
# that can occur when the slider was very close to the upcoming outcome.
LR_CLEAN_MIN = -0.6
LR_CLEAN_MAX = +1.3 

# This choice of window width was determined by computing the empirical
# distribution of the num. of outcomes (=trials) between two change points
# in our dataset of sequences, and taking a value close to the mean and median
# of this distribution (see below for mean and median).
CP_DISTANCE_WINDOW_WIDTH_BY_TASK = {
    "ada-pos": 10, # choice based on the empirical distance distribution
                   # median: 8
                   # mean: 10.6
    "ada-prob": 18, # choice based on the empirical distance distribution
                    # median: 16
                    # mean: 19.0
}
CP_DISTANCE_WINDOW_MIN = -2

# Value where the estimate slider is positioned at the start of each session
ESTIMATE_PRIOR = 0.5

# Legacy code: We considered different possible ways of measuring prior uncertainty.
# In the paper, we use the SD of the normative model's distribution about
# the hidden magnitude/probability just prior to having observed the last outcome.
CONF_TYPE_BY_TASK = {"ada-pos": "prev-conf-pred", "ada-prob": "prev-conf"}
UNCERTAINTY_UNIT = "sd"

# Stimulus Onset Asyncrony:
# This is the time inverval in the task between
# the onset of one outcome stimulus and the onset of the next outcome stimulus.
SOA_MS = 1500
SOA_S = SOA_MS / 1e3

def create_dir_if_needed(d):
    if not op.exists(d):
        os.makedirs(d, exist_ok=True)

def get_path(dir, basename, ext="png"):
    fname = f"{basename}.{ext}"
    return op.join(dir, fname)

def get_output_dir_for_data_file(data_file, create_if_needed=False,
    is_wip=False):
    output_dir = op.join(OUTPUT_BASE_DIR, op.splitext(op.basename(data_file))[0])
    if is_wip:
        output_dir = op.join(output_dir, OUTPUT_WIP_DIR)
    if create_if_needed:
        create_dir_if_needed(output_dir)
    return output_dir

def get_output_dir_for_group_def_file(group_def_file, create_if_needed=False,
    is_wip=False):
    group_id = get_group_id(group_def_file)
    output_dir = op.join(OUTPUT_GROUP_BASE_DIR, group_id)
    if is_wip:
        output_dir = op.join(output_dir, OUTPUT_WIP_DIR)
    if create_if_needed:
        create_dir_if_needed(output_dir)
    return output_dir

def get_subject_data_files_from_group_def(group_def_file):
    df = pd.read_csv(group_def_file)
    data_files = df["data_file"].tolist()
    return data_files

def get_group_id(group_def_file):
    name = op.splitext(op.basename(group_def_file))[0]
    prefix_to_remove = "group_def_"
    if name.startswith(prefix_to_remove):
        name = name[len(prefix_to_remove):]
    return name

def load_data(data_file):
    df = pd.read_csv(data_file)
    if df.shape[0] == 1:
        # current data structure version:
        # the data file is a csv table with a single column and row
        # containing the whole data for the given study run as one json string
        data = json.loads(df.iloc[0, 0])
    else:
        # previous data structure version:
        # the data file is a csv table with a single column and n rows
        # with one row per task session
        sessions = [json.loads(jsonString)
            for jsonString in df['taskSessionDataJSON']]
        # convert this data structure into the new structure
        data = {"taskSessionDataArray": sessions}
    return data

def get_task_to_subject_sessions(data_files):
    data_s = [load_data(data_file) for data_file in data_files]
    n_tasks = data_s[0].get("nTasks", 1)
    if n_tasks == 1:
        task = data_s[0]["taskSessionDataArray"][0]["taskName"]
        return {task: [data["taskSessionDataArray"] for data in data_s]}
    # else:
    tasks = [sessions[0]["taskName"] for sessions in data_s[0]["taskSessionDataArrays"]]
    task_to_sub_sessions = {}
    for task in tasks:
        task_to_sub_sessions[task] = []
        for data in data_s:
            for sessions in data["taskSessionDataArrays"]:
                if sessions[0]["taskName"] == task:
                    task_to_sub_sessions[task] += [sessions]
                    break
    return task_to_sub_sessions
        
def get_sessions_per_task(data):
    n_tasks = data.get("nTasks", 1)
    return (data["taskSessionDataArrays"] if n_tasks > 1
        else [data["taskSessionDataArray"]])

def make_sessions_by_task_from_seq_data(seq_data_file):
    df_all_tasks = pd.read_csv(seq_data_file)
    tasks = df_all_tasks["taskName"].unique()
    sessions_by_task = {}
    for task in tasks:
        df = df_all_tasks[df_all_tasks["taskName"] == task]
        p_c = df["pC"].iloc[0]
        n_trials = df["nTrials"].iloc[0]
        if task == "ada-prob":
            p1_min = df["p1Min"].iloc[0]
            p1_max = df["p1Max"].iloc[0]
            change_key = "didP1Change"
        elif task == "ada-pos":
            mean_min = df["meanMin"].iloc[0]
            mean_max = df["meanMax"].iloc[0]
            std_gen = df["std"].iloc[0]
            change_key = "didMeanChange"
        min_run_length = df["minRunLength"].iloc[0]
        max_run_length = df["maxRunLength"].iloc[0]

        all_sessions = []
        for i_sess, df_sess in df.groupby("sessionIndex"):
            session = {
                "taskName": task,
                "nTrials": n_trials,
                "sequencePC": p_c,
            }
            outcomes = df_sess["outcome"].to_numpy()
            session["outcome"] = outcomes
            session[change_key] = df_sess[change_key].to_numpy(dtype=bool)
            if task == "ada-prob":
                session["sequenceP1Min"] = p1_min
                session["sequenceP1Max"] = p1_max
                estimates = model.ideal_learner_prob_estimates_from_outcomes(outcomes,
                    p_c, p1_min=p1_min, p1_max=p1_max)
            elif task == "ada-pos":
                session["sequenceMeanMin"] = mean_min
                session["sequenceMeanMax"] = mean_max
                session["sequenceStd"] = std_gen
                estimates = model.reduced_learner_pos_estimates_from_outcomes(
                    outcomes, p_c=p_c, std_gen=std_gen)
            session["estimate"] = estimates # simulate as if subjects were ideal observers
            all_sessions += [session]
        sessions_by_task[task] = all_sessions
    return sessions_by_task

def get_sessions_event_segments_file_for_data_file(data_file,
    create_dir_if_needed=False,
    ext="pkl"):
    odir = get_output_dir_for_data_file(data_file, create_if_needed=create_dir_if_needed)
    fname = f"sessions_event_segments.{ext}"
    return op.join(odir, fname)

def save_sessions_event_segments(sessions_event_segments, file):
    with open(file, 'wb') as f:
        pickle.dump(sessions_event_segments, f)

def load_sessions_event_segments(file):
    with open(file, 'rb') as f:
        sessions_event_segments = pickle.load(f)
    return sessions_event_segments

def load_sessions_event_segments_with_data_file(data_file):
    file = get_sessions_event_segments_file_for_data_file(data_file)
    return load_sessions_event_segments(file)

def all_sessions_completed(sessions):
    return np.all(np.array([session["endTime"] > 0 for session in sessions], dtype=bool))

def get_data_arrays_from_sessions(sessions, keys=[]):
    """
    Keys for variables not derived from a model learner:
    - 'taskName': the name of the task ('ada-prob' or 'ada-pos')
    - 'nTrials': the num. of trials per session
    - 'outcome': the sequence of outcomes in each session
    - 'scoreFrac': subject's score in each sesion
    - 'estimate': subject's estimates
    - 'update': subject's updates (0 when no update)
    - 'error': difference between outcome and subject's prev. estimate
    - 'learning_rate': subject's learning rates (clean)
    - 'learning_rate_raw': subject's learning rates (raw)
    - 'was_estimate_updated': whether they updated their estimate in a given trial
    - 'outcome-error-magnitude': magnitude of the difference between each outcome
                                and subject's estimate before the outcome
    - 'p1' (for Ada-Prob only): the generative probability
    - 'mean' (for Ada-Pos only): the generative mean
    - 'didP1Change'/'didMeanChange': True iff a change point occurred
                                    (always True at the very first trial)
    - 'sequencePC': the generative change point probability
    - 'sequenceP1Min', 'sequenceP1Max' (for Ada-Prob only): the bounds of the possible generative probability
    - 'sequenceMeanMin', 'sequenceMeanMax' (for Ada-Prob only): the bounds of the possible generative probability
    - 'sequenceStd' (for Ada-Pos only): the generative standard deviation
    - 'sequenceIdx': the index of the outcome sequence in the seq-data dataset
    
    Keys for model learner-derived variables:
    - 'model_estimate': model's estimate
    - 'model_learning_rate': model's learning rate
    - 'model_uncertainty': model's uncertainty associated with the previous estimate
    - 'model_relative-uncertainty': model's relative uncertainty associated
       with the previous estimate, as defined by Nassar et al. (2010, 2012)
    - 'model_cpp': change point probability afforded by the last outcome,
                   as defined by Nassar et al. (2010, 2012)
    - 'model_outcome-error-magnitude': magnitude of the difference between each outcome
                                       and model's estimate before the outcome
    - 'model_prev-conf' or 'model_prev-conf-pred': model's confidence at the previous trial
                                                  derived from the posterior distribution
                                                  in the generative value
                                                  on the same trial (prev-conf)
                                                  or on the next trial (prev-conf-pred)
    """
    keys_set = set(keys)

    n_sessions = len(sessions)
    n_trials = sessions[0].get("nTrials")
    task_name = sessions[0]["taskName"]

    # Construct the set of all variable keys we need to retrieve from
    # the raw data
    all_raw_data_trial_keys = ["estimate", "outcome", "p1", "mean", "didP1Change", "didMeanChange"]
    all_raw_data_sess_keys = ["scoreFrac", "sequenceIdx"]
    all_raw_data_task_keys = ["taskName", "nTrials", "sequencePC", "sequenceP1Min", "sequenceP1Max",
    "sequenceMeanMin", "sequenceMeanMax", "sequenceStd"]
    all_model_trial_keys = ["model_estimate", 
        "model_uncertainty", "model_relative-uncertainty",
        "model_prev-conf", "model_prev-conf-pred",
        "model_outcome-error-magnitude", "model_cpp"]
    raw_data_trial_keys = keys_set & set(all_raw_data_trial_keys)
    raw_data_sess_keys = keys_set & set(all_raw_data_sess_keys)
    raw_data_task_keys = keys_set & set(all_raw_data_task_keys)
    model_trial_keys = keys_set & set(all_model_trial_keys)

    if (('learning_rate' in keys)
        or ('update' in keys)
        or ('error' in keys)
        or ('learning_rate_raw' in keys)
        or ('outcome-error-magnitude' in keys)
        or ('was_estimate_updated' in keys_set)):
        raw_data_trial_keys.add("estimate")
        raw_data_trial_keys.add("outcome")
    if (('model_learning_rate' in keys)
        or ('model_outcome-error-magnitude' in keys)):
        model_trial_keys.add("model_estimate")
        model_trial_keys.add("outcome")

    if (len(model_trial_keys) > 0):
        raw_data_trial_keys.add("outcome")
        raw_data_task_keys.add("sequencePC")
        if task_name == "ada-prob":
            raw_data_task_keys.add("sequenceP1Min")
            raw_data_task_keys.add("sequenceP1Max")
        elif task_name == "ada-pos":
            raw_data_task_keys.add("sequenceMeanMin")
            raw_data_task_keys.add("sequenceMeanMax")
            raw_data_task_keys.add("sequenceStd")

    # Retrieve variables from raw data
    arrays = {}
    for key in raw_data_task_keys:
        arrays[key] = sessions[0][key]
    for key in raw_data_sess_keys:
        if key == "sequenceIdx":
            arrays[key] = np.empty((n_sessions), dtype=int)
        else:
            arrays[key] = np.empty((n_sessions))
    for key in raw_data_trial_keys:
        if (key == "didP1Change"
            or key == "didMeanChange"):
            arrays[key] = np.empty((n_sessions, n_trials), dtype=bool)
        else:
            arrays[key] = np.empty((n_sessions, n_trials))
    for i in range(n_sessions):
        for key in (raw_data_trial_keys | raw_data_sess_keys):
            arrays[key][i] = sessions[i][key]
    
    # Compute variables derived from raw data using model learner
    for key in model_trial_keys:
        if (key == "model_estimate"):
            if task_name == "ada-prob":
                arrays[key] = model.ideal_learner_prob_estimates_from_outcomes(
                    arrays["outcome"], p_c=arrays["sequencePC"],
                    p1_min=arrays["sequenceP1Min"], p1_max=arrays["sequenceP1Max"])
            elif task_name == "ada-pos":
                arrays[key] = model.reduced_learner_pos_estimates_from_outcomes(
                    arrays["outcome"], p_c=arrays["sequencePC"], std_gen=arrays["sequenceStd"])
        elif (key == "model_uncertainty"):
            conf_type = CONF_TYPE_BY_TASK[task_name]
            if task_name == "ada-prob":
                confs = model.ideal_learner_prob_confidence_from_outcomes(
                    arrays["outcome"], p_c=arrays["sequencePC"], conf_type=conf_type,
                    p1_min=arrays["sequenceP1Min"], p1_max=arrays["sequenceP1Max"])
            elif task_name == "ada-pos":
                inf_dict = model.reduced_learner_pos_inference_from_outcomes(
                    arrays["outcome"], p_c=arrays["sequencePC"], std_gen=arrays["sequenceStd"])
                confs = model.reduced_learner_pos_confidence_with_type(
                    inf_dict, conf_type="prev-conf-pred", std_gen=arrays["sequenceStd"])
            arrays[key] = uncertainty_from_confidence(confs)
        elif (key == "model_relative-uncertainty"):
            conf_type = CONF_TYPE_BY_TASK[task_name]
            if task_name == "ada-prob":
                assert False, "relative uncertainty not implemented for ada-prob"
            elif task_name == "ada-pos":
                inf_dict = model.reduced_learner_pos_inference_from_outcomes(
                    arrays["outcome"], p_c=arrays["sequencePC"], std_gen=arrays["sequenceStd"])
                rus = model.reduced_learner_pos_confidence_with_type(
                    inf_dict, conf_type="relative-prev-uncertainty-pred", std_gen=arrays["sequenceStd"])
            arrays[key] = rus
        elif (key == "model_prev-conf"):
            if task_name == "ada-prob":
                arrays[key] = model.ideal_learner_prob_confidence_from_outcomes(
                    arrays["outcome"], p_c=arrays["sequencePC"], conf_type="prev-conf",
                    p1_min=arrays["sequenceP1Min"], p1_max=arrays["sequenceP1Max"])
        elif (key == "model_prev-conf-pred"):
            if task_name == "ada-prob":
                arrays[key] = model.ideal_learner_prob_confidence_from_outcomes(
                    arrays["outcome"], p_c=arrays["sequencePC"], conf_type="prev-conf-pred",
                    p1_min=arrays["sequenceP1Min"], p1_max=arrays["sequenceP1Max"])
            if (task_name == "ada-pos"):
                inf_dict = model.reduced_learner_pos_inference_from_outcomes(
                    arrays["outcome"], p_c=arrays["sequencePC"], std_gen=arrays["sequenceStd"])
                arrays[key] = model.reduced_learner_pos_confidence_with_type(
                    inf_dict, conf_type="prev-conf-pred", std_gen=arrays["sequenceStd"])
        elif (key == "model_cpp"):
            if task_name == "ada-prob":
                arrays[key] = model.ideal_learner_prob_cpp(
                    arrays["outcome"], p_c=arrays["sequencePC"],
                    p1_min=arrays["sequenceP1Min"], p1_max=arrays["sequenceP1Max"])
            elif task_name == "ada-pos":
                arrays[key] = model.reduced_learner_pos_cpp(
                    arrays["outcome"], p_c=arrays["sequencePC"], std_gen=arrays["sequenceStd"])

    # Compute variables derived from outcomes and subject or model estimates
    if (('learning_rate' in keys_set)
        or ('update' in keys_set)
        or ('error' in keys_set)
        or ('learning_rate_raw' in keys_set)
        or ('outcome-error-magnitude' in keys_set)
        or ('was_estimate_updated' in keys_set)):
        measures = estimate_update_measures(arrays["estimate"], arrays["outcome"])
        arrays["outcome-error-magnitude"] = np.abs(measures["errors"])
        arrays["update"] = measures["updates"]
        arrays["error"] = measures["errors"]
        arrays["learning_rate"] = measures['learning_rates']
        arrays["learning_rate_raw"] = measures['learning_rates_raw']
        arrays["was_estimate_updated"] = measures['was_estimate_updated']
    if (('model_learning_rate' in keys_set)
        or ('model_outcome-error-magnitude' in keys_set)):
        measures = estimate_update_measures(arrays["model_estimate"], arrays["outcome"])
        arrays["model_outcome-error-magnitude"] = np.abs(measures["errors"])
        arrays["model_learning_rate"] = measures['learning_rates']

    # Filter from the data only the keys that were passed in arguments
    arrays = {k: v for (k, v) in arrays.items() if k in keys_set}

    return arrays

def estimate_update_measures_from_sessions(sessions, estimate_prior=ESTIMATE_PRIOR):
    # assert all_sessions_completed(sessions)
    arrays = get_data_arrays_from_sessions(sessions, ["estimate", "outcome"])
    return estimate_update_measures(arrays["estimate"], arrays["outcome"],
        estimate_prior=estimate_prior)

def estimate_update_measures(estimates, outcomes, estimate_prior=ESTIMATE_PRIOR):
    # assert (estimates.shape == outcomes.shape)
    # compute measures on 2d arrays of shape (n_sessions, n_trials)
    if estimates.ndim == 1:
        estimates = estimates[np.newaxis, :]
        outcomes = outcomes[np.newaxis, :]
        
    estimates_with_prior = get_values_with_prior(estimates, estimate_prior)

    estimate_updates = (estimates_with_prior[:, 1:] - estimates_with_prior[:, 0:-1])
    estimate_errors = (outcomes - estimates_with_prior[:, 0:-1])
    lrs_raw = estimate_updates / estimate_errors
    lrs = get_clean_learning_rates(lrs_raw)
    was_estimate_updated = ~np.isclose(estimate_updates, 0.)
    return dict(updates=estimate_updates,
        errors=estimate_errors,
        learning_rates=lrs,
        learning_rates_raw=lrs_raw,
        was_estimate_updated=was_estimate_updated)

def uncertainty_from_confidence(conf, unit=UNCERTAINTY_UNIT, cpp=None):
    if unit == "sd":
        return np.exp(-conf)
    elif unit == "log sd":
        return -conf
    elif unit == "-log sd":
        return conf
    elif unit == "sd*(1-cpp)" and (cpp is not None):
        return np.exp(-conf) *  (1 - cpp)

def get_values_with_prior(values, prior):
    if values.ndim == 2:
        return np.insert(values, 0, prior, axis=1)
    elif values.ndim == 1:
        return np.insert(values, 0, prior, axis=0)

def get_prev_estimates(estimates, estimate_prior=ESTIMATE_PRIOR):
    estimates_with_prior = get_values_with_prior(estimates, prior=estimate_prior)
    if estimates.ndim == 2:
        return estimates_with_prior[:, :-1] #np.insert(estimates[:, :-1], 0, estimate_prior, axis=1)
    elif estimates.ndim == 1:
        return estimates_with_prior[:-1] #np.insert(estimates[:-1], 0, estimate_prior, axis=0)

def get_clean_learning_rates(lrs_raw):
    # places NaN values where learning rates is above or below the clean lr bounds
    return np.where((lrs_raw >= LR_CLEAN_MIN) & (lrs_raw <= LR_CLEAN_MAX),
        lrs_raw, np.nan)

def get_change_point_probability(session):
    if "sequencePC" in session:
        return session["sequencePC"]
    elif "pC" in session:
        return session["pC"]

def get_did_changes(session):
    task_name = session["taskName"]
    if task_name == "ada-prob":
        change_key = 'didP1Change'
    elif task_name == "ada-pos":
        change_key = 'didMeanChange'
    return session[change_key]

def get_generative_values(session):
    task_name = session["taskName"]
    gen_key = get_gen_key_for_task(task_name)
    return session[gen_key]

def get_gen_key_for_task(task_name):
    if task_name == "ada-prob":
        return 'p1'
    elif task_name == "ada-pos":
        return 'mean'

def get_hidvar_bounds(session):
    task_name = session["taskName"]
    if task_name == "ada-prob":
        key_min = "sequenceP1Min"
        key_max = "sequenceP1Max"
    elif task_name == "ada-pos":
        key_min = "sequenceMeanMin"
        key_max = "sequenceMeanMax"
    return (session[key_min], session[key_max])

def get_std_gen(session):
    task_name = session["taskName"]
    if task_name == "ada-pos":
        return session["sequenceStd"]

def get_sequence_data_file(session,
    ext="csv"):
    fileJS = session["sequenceDataFile"]
    file = op.splitext(fileJS)[0]
    file += f".{ext}"
    return file

def get_absolute_errors(session, custom_estimates=None):
    if custom_estimates is None:
        estimates = np.array(session["estimate"])
    else:
        estimates = custom_estimates
    gens = np.array(get_generative_values(session))
    return np.abs(estimates - gens)

def get_absolute_error_upper_bound(session):
    task_name = session["taskName"]
    if task_name == "ada-prob":
        return 0.5
    elif task_name == "ada-pos":
        return session["sequenceStd"] * 2

def get_window_around_cp_for_sessions(sessions):
    task = sessions[0]["taskName"]
    return get_window_around_cp(task)

def get_window_around_cp(task, w_min=CP_DISTANCE_WINDOW_MIN):
    # Note that window = [m, m+1, …, m+width]
    # so the length of the window is width+1.
    win_width = CP_DISTANCE_WINDOW_WIDTH_BY_TASK[task]
    return np.arange(win_width+1) + w_min

def get_outcomes_after_cp(win):
    return win+1

def get_sec_after_cp(win):
    return (win+1) * SOA_S

def aggregate_values_in_window_around_cp(values, window_around_cp, sessions,
    do_ignore_start_cp=False, discard_nan=True):
    """
    Aggregate the values of all trials that fall into the provided window
    around a change point by their distance to the change point.
    Returns a list of lists of values where each list corresponds to one distance
    to the change point within the provided window.
    """
    n_trials = sessions[0]["nTrials"]
    win_around_cp_min = window_around_cp.min()
    win_around_cp_max = window_around_cp.max()
    values_in_win_around_cp = [[] for t in window_around_cp]
    for i_sess, session in enumerate(sessions):
        # Iterate over change points
        did_changes = get_did_changes(session)
        if do_ignore_start_cp:
            did_changes[0] = False
        cp_trial_indices = np.arange(n_trials)[did_changes]
        dist_between_cps = np.diff(cp_trial_indices)
        for i_cp, trial_cp in enumerate(cp_trial_indices):
            # Compute the window of trials to take around this change point
            # as the intersection of
            # - the defined window to aggregate over
            # - the window of trials between the previous and the next change point
            # (or starting at the first trial of the session if there is no previous c.p.,
            # and ending at the last trial of the session if there is no next c.p.)
            if i_cp > 0:
                dist_to_prev_cp = dist_between_cps[i_cp-1]
                trial_win_min = max(trial_cp+win_around_cp_min,
                    trial_cp-dist_to_prev_cp+1)
            else:
                trial_win_min = max(trial_cp+win_around_cp_min, 0)
            if i_cp < len(dist_between_cps):
                dist_to_next_cp = dist_between_cps[i_cp]
                trial_win_max = min(trial_cp+win_around_cp_max,
                    trial_cp+dist_to_next_cp-1)
            else:
                trial_win_max = min(trial_cp+win_around_cp_max, n_trials-1)
            for trial in range(trial_win_min, trial_win_max+1):
                v = values[i_sess][trial]
                if not (discard_nan and np.isnan(v)):
                    t_from_cp = trial-trial_cp
                    i_win = t_from_cp - win_around_cp_min
                    values_in_win_around_cp[i_win] += [v]
    return values_in_win_around_cp

def compute_lrs_around_cp(sessions, window_around_cp,
    estimate_key="estimate", # key specifying which estimates to retrieve
                             # using get_data_array_from_sessions().
                             # by default these are the subject's estimates.
    estimate_fun=None, # function called to compute estimates from the observations.
                       # mutually exclusive with 'estimate_key'.
                       # the function signature should be take an array of
                       # observation sequences as input and return an array
                       # of estimate sequences as output.
    use_only_trials_with_update=False,
    do_ignore_start_cp=False,
    do_assert=False):
    if do_assert:
        assert all_sessions_completed(sessions)
    keys = ["outcome"]
    if estimate_key is not None:
        keys += [estimate_key]
    if use_only_trials_with_update:
        keys += ["was_estimate_updated"]
    # Compute learning rates for all trials
    arrays = get_data_arrays_from_sessions(sessions, keys=keys)
    outcome = arrays["outcome"]
    if estimate_fun is not None:
        estimate = estimate_fun(outcome)
    else:
        estimate = arrays[estimate_key]
    lrs = estimate_update_measures(estimate, outcome)["learning_rates"]
    # Discard trials where the subject's estimate was not updated, if needed
    if use_only_trials_with_update:
        lrs = np.where(arrays["was_estimate_updated"], lrs, np.nan)
    # Aggregate the learning rates of all trial per trial distance in the window
    return aggregate_values_in_window_around_cp(lrs, window_around_cp, sessions,
        do_ignore_start_cp=do_ignore_start_cp)

def compute_update_freq_magn_around_cp(sessions, window_around_cp):
    assert all_sessions_completed(sessions)
    # Compute whether subject updated for each trial
    # and the magnitude of the update in case of they did
    arrays = get_data_arrays_from_sessions(sessions, keys=[
        "was_estimate_updated", "update"])
    was_estimate_updated = arrays["was_estimate_updated"]
    update_magns[i, :] = np.where(was_estimate_updated,
        arrays["update"],
        np.nan) # replace zero with nan in order to ignore this value when computing mean magnitude
    # Aggregate the occurrences and magnitude of updates
    # of all trials per trial distance in the window
    # (ignoring update magnitude when there was no update)
    update_flags_around_cp = aggregate_values_in_window_around_cp(was_estimate_updated,
        window_around_cp, sessions)
    update_magns_around_cp = aggregate_values_in_window_around_cp(update_magns,
        window_around_cp, sessions)
    updatefreq_around_cp = np.array([np.mean(update_flags) for update_flags in update_flags_around_cp])
    updatemagn_around_cp = np.array([np.mean(update_magns) for update_magns in update_magns_around_cp])
    return updatefreq_around_cp, updatemagn_around_cp


def compute_response_latencies(sessions, sessions_event_segments):
    n_sessions = len(sessions)
    n_trials = sessions[0]["nTrials"]
    response_latencies = np.empty((n_sessions, n_trials))
    for i in range(n_sessions):
        response_latencies[i, :] = compute_response_latencies(
            sessions_event_segments[i], n_trials)
    return response_latencies

def compute_response_latencies_around_cp(sessions, window_around_cp,
    sessions_event_segments):
    # Compute distance to change point and response latencies for all trial
    n_sessions = len(sessions)
    n_trials = sessions[0]["nTrials"]
    response_latencies = np.empty((n_sessions, n_trials))
    for i in range(n_sessions):
        response_latencies[i, :] = compute_response_latencies(
            sessions_event_segments[i], n_trials)
    # Aggregate the response latencies of all trial per trial distance in the window
    # (ignoring trials where we did not get a response latency)
    rls_around_cp = aggregate_values_in_window_around_cp(response_latencies,
        window_around_cp, sessions)
    return rls_around_cp

def compute_dist_between_cps(session):
    cps = get_session_change_point_indices(session)
    return np.diff(cps)

def compute_trial_dist_to_last_cp(session):
    # cp stands for "change point"
    cps = get_session_change_point_indices(session)
    n_trials = session["nTrials"]
    n_cps = cps.shape[0]
    dist_to_cp = np.empty(n_trials, dtype=int)
    dist_to_cp[:] = np.iinfo(np.int32).max
    for i_cp in range(n_cps):
        last_cp = cps[i_cp]
        if i_cp < (n_cps - 1):
            next_cp = cps[i_cp+1]
            dist_to_last = np.arange(next_cp - last_cp)
            dist_to_cp[last_cp:next_cp] = dist_to_last
        else:
            dist_to_last = np.arange(n_trials - last_cp)
            dist_to_cp[last_cp:n_trials] = dist_to_last
    return dist_to_cp

def get_session_change_point_indices(session):
    did_changes = get_did_changes(session)
    n_trials = session["nTrials"]
    cps = np.arange(n_trials)[did_changes]
    return cps

def aggregate_values_in_baseline_trials(values, baseline_trials, discard_nan=True):
    assert values.ndim == 2
    n_sessions = values.shape[0]
    vals_baseline = []
    for i_sess in range(n_sessions):
        vals_i = values[i_sess, baseline_trials["index"][i_sess]]
        if discard_nan:
            vals_i = vals_i[~np.isnan(vals_i)]
        vals_baseline += [vals_i]
    return np.concatenate(vals_baseline)

def compute_baseline_trials(sessions,
    method="quantile_across_sessions", q_baseline=0.2):
    n_sessions = len(sessions)
    n_trials = sessions[0]["nTrials"]
    trial_dists_to_last_cp = np.empty((n_sessions, n_trials), dtype=int)
    for i_sess, session in enumerate(sessions):
        trial_dists_to_last_cp[i_sess, :] = compute_trial_dist_to_last_cp(session)
    if method == "quantile_across_sessions":
        # compute the quantile distance to last c.p. across all sessions,
        # on a flattened version of the array of trial distances.
        # each session will have the same minimum distance threshold for the baseline,
        # but a different num. of baseline trials.
        thresh_dist = np.quantile(trial_dists_to_last_cp, (1-q_baseline))
        baseline_trial_indices = [
            np.nonzero(trial_dists_to_last_cp[i_sess, :] > thresh_dist)[0]
            for i_sess in range(n_sessions)]
    elif method == "quantile_within_session":
        # compute the quantile distance to last c.p. within each session.
        # each session will have the same num. of baseline trials, but
        # a different minimum distance threshold for the baseline.
        thresh_dist = np.quantile(trial_dists_to_last_cp, (1-q_baseline),
            axis=1, keepdims=True)
        baseline_trial_indices = [
            np.nonzero(trial_dists_to_last_cp[i_sess, :] > thresh_dist[i_sess, :])[0]
            for i_sess in range(n_sessions)]

    baseline_trial_dists_to_last_cp = [
            trial_dists_to_last_cp[i_sess, :][baseline_trial_indices[i_sess]]
            for i_sess in range(n_sessions)]
    baseline_trials = {
        "index": baseline_trial_indices,
        "dist_to_last_cp": baseline_trial_dists_to_last_cp
    }

    return baseline_trials

def compute_response_latencies(event_segments, n_trials):
    response_latencies = np.empty(n_trials)
    response_latencies[:] = np.nan
    trial_segment = None
    did_find_first_movement_for_trial = False
    for segment in event_segments:
        if (segment["event_type"] == "trial"):
            trial_segment = segment
            did_find_first_movement_for_trial = False
        elif ((segment["event_type"] == "slider_movement")
            and (trial_segment is not None)):
            if not did_find_first_movement_for_trial:
                response_latency = (segment["start_time"] - trial_segment["start_time"])
                response_latencies[trial_segment["index"]] = response_latency
                did_find_first_movement_for_trial = True
    return response_latencies

def get_bin_mask(x, bin_edges, i_bin):
    n_bins = len(bin_edges) - 1
    if i_bin < (n_bins - 1):
        return ((x >= bin_edges[i_bin]) & (x < bin_edges[i_bin+1]))
    else:
        return ((x >= bin_edges[i_bin]) & (x <= bin_edges[i_bin+1]))

#
### ANALYSES OF SLIDER MOVEMENTS
#

# Define minimum change in estimate value to consider that the slider has been moved
# as a change of 1 device pixel, with the defined maximum device-to-css pixel ratio.
SLIDER_TRACK_WIDTH_CSS_PX = 640
MAXIMUM_DEVICE_PX_TO_CSS_PX_RATIO = 2
SLIDER_MOVE_MIN_ESTIMATE_CHANGE = (1
    / (MAXIMUM_DEVICE_PX_TO_CSS_PX_RATIO * SLIDER_TRACK_WIDTH_CSS_PX))

def compute_slidermove_instant_events(session,
    estimate_prior=ESTIMATE_PRIOR,
    min_estimate_change=SLIDER_MOVE_MIN_ESTIMATE_CHANGE):
    # Retrieve all mouse move events of the session.
    mousemove_events = session["events"]["estimateUpdate"]
    # Process this sequence to retain only events where the estimate has actually changed.
    estimate = estimate_prior
    slidermove_instant_events = []
    for event in mousemove_events:
        new_estimate = event[1]
        if abs(new_estimate - estimate) >= SLIDER_MOVE_MIN_ESTIMATE_CHANGE:
            slidermove_instant_events += [{
                "time": event[0],
                "estimate": new_estimate }]
            estimate = new_estimate
    return slidermove_instant_events

# Maximum time interval between slider movements within the same estimate update
# Informed by the histogram of inter-slidermove-instant-intervals
# Informed by refs:
# - Der, G., & Deary, I. J. (2006). Age and sex differences in reaction time
#   in adulthood… . Psychology and aging, 21(1), 62.
#    7,216 completed a simple reaction time task, in subjects of age 20 to 40,
#    mean RT ranged in [289ms, 323ms]
#    and mean intraindividual SD ranged in [71ms, 83ms]
# - Jaśkowski, P. (1983). Distribution of the human reaction time measurements. Acta Neurobiologiae Experimentalis.
#   https://www.ane.pl/pdf/4320.pdf: 
#   Argmax of the distribution of reaction time in a simple visual reaction time
#   task with a clearly visible stimulus was 260ms
# - Kosinski, R. J. (2008). A literature review on reaction time. Clemson University, 10(1), 337-344.
#   "…mean visual reaction times being 180-200 msec (Galton, 1899;
#    Woodworth and Schlosberg, 1954; Fieandt et al., 1956; Welford, 1980;
#    Brebner and Welford, 1980)."
MAX_INTER_SLIDERMOVE_INTERVAL_WITHIN_ONE_UPDATE_MS = 150 # 260

# Do not consider changes of estimate that are below these value as real slider movements
SLIDER_MOVEMENT_MIN_ESTIMATE_CHANGE = 0.005

def compute_event_segments(session):
    """Return the event segments (trials and slider movements, and one
    time segment representing the whole session) sorted by their start time"""
    slider_movement_segments = compute_slider_movement_segments(session)
    # print("slider_movement_segments", slider_movement_segments)
    trial_segments = compute_trial_segments(session)
    # print("trial_segments", trial_segments)
    session_segment = {
        "start_time": session["events"]["sessionStart"][0][0],
        "end_time": session["events"]["sessionEnd"][0][0],
        "event_type": "session"}
    # print("session_segment", session_segment)
    segments = sorted([session_segment] + trial_segments + slider_movement_segments,
        key=(lambda s: s["start_time"]))
    return segments

def compute_trial_segments(session):
    trial_segments = []
    last_trial_start_ev = session["events"]["trialStart"][0]
    for trial_start_ev in session["events"]["trialStart"][1:]:
        trial_segments += [ make_trial_segment(
            last_trial_start_ev[0], trial_start_ev[0], last_trial_start_ev[1])]
        last_trial_start_ev = trial_start_ev
    # last trial
    session_end_time = session["events"]["sessionEnd"][0][0]
    trial_segments += [ make_trial_segment(
            last_trial_start_ev[0], session_end_time, last_trial_start_ev[1])]
    return trial_segments

def compute_slider_movement_segments(session,
    slider_movement_min_estimate_change=SLIDER_MOVEMENT_MIN_ESTIMATE_CHANGE,
    segmentation_threshold=MAX_INTER_SLIDERMOVE_INTERVAL_WITHIN_ONE_UPDATE_MS,
    estimate_prior=ESTIMATE_PRIOR):
    slidermove_instant_events = compute_slidermove_instant_events(session)
    slider_movement_segments = []
    current_segment = None
    start_estimate_for_next_segment = estimate_prior
    for ev in slidermove_instant_events:
        if (current_segment is None):
            # make the initial segment
            current_segment = make_new_slider_movement_segment(ev, start_estimate_for_next_segment)
        elif ((ev["time"] - last_ev["time"])
            > MAX_INTER_SLIDERMOVE_INTERVAL_WITHIN_ONE_UPDATE_MS):
            # close the current segment
            if (current_segment is not None):
                close_slider_movement_segment(current_segment, last_ev)
                # add it to the list if the change of estimate is above minimum
                if (np.abs(current_segment["end_estimate"] - current_segment["start_estimate"])
                    > slider_movement_min_estimate_change):
                    slider_movement_segments += [current_segment]
                    start_estimate_for_next_segment = current_segment["end_estimate"]
            # create a new segment
            new_segment = make_new_slider_movement_segment(ev, start_estimate_for_next_segment)
            current_segment = new_segment
        else:
            # extend the current segment
            extend_slider_movement_segment(current_segment, ev)

        last_ev = ev

    # close the last segment
    close_slider_movement_segment(current_segment, last_ev)
    # add it to the list if the change of estimate is above minimum
    if (np.abs(current_segment["end_estimate"] - current_segment["start_estimate"])
        > slider_movement_min_estimate_change):
        slider_movement_segments += [current_segment]

    return slider_movement_segments

def make_trial_segment(start_time, end_time, index):
    return {
        "start_time": start_time,
        "end_time": end_time,
        "event_type": "trial",
        "index": index}

def make_new_slider_movement_segment(start_ev, start_estimate):
    return {
        "start_time": start_ev["time"],
        "end_time": None,
        "event_type": "slider_movement",
        "start_estimate": start_estimate,
        "end_estimate": None,
        "estimate_trajectory": [(start_ev["time"], start_ev["estimate"])]
    }

def extend_slider_movement_segment(segment, ev):
    segment["estimate_trajectory"] += [(ev["time"], ev["estimate"])]

def close_slider_movement_segment(segment, last_ev):
    segment["end_time"] = last_ev["time"]
    segment["end_estimate"] = last_ev["estimate"]

