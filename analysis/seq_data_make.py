"""
Utilities to make sequence data for the adaptive learning tasks.
"""

import argparse
import json
import model_learner as model
import numpy as np
import pandas as pd
import random as rd

N_SESSIONS = 1
N_TRIALS = 75
P_C = 1/20
P1_MIN = 0.1
P1_MAX = 0.9
MIN_RUN_LENGTH = 6
MAX_RUN_LENGTH = 250
MIN_ODD_CHANGE = 4

SCORE_MIN_MSE = (0.25) ** 2

GAUSSIAN_P_C = 1/10 # to match Nassar et al. (2010)
GAUSSIAN_STD = 10/300
GAUSSIAN_MIN_RUN_LENGTH = 3
GAUSSIAN_MAX_RUN_LENGTH = None
GAUSSIAN_MEAN_MIN = 0.1
GAUSSIAN_MEAN_MAX = 0.9
GAUSSIAN_OUTCOME_MIN = 0
GAUSSIAN_OUTCOME_MAX = 1
GAUSSIAN_MIN_ODD_CHANGE = 1

# Adapted from NeuralProb/utilities/utils.py
def generate_p1s_sequence(n_trials, p_c,
    p1_min=P1_MIN, # minimum value for probabilities
    p1_max=P1_MAX, # maximum value for probabilities
    min_run_length=MIN_RUN_LENGTH, # minimum length for a stable period
                                   # /!\ NOTE: this minimum does not effectively apply
                                   # for the last stable period at the end of the sequence
                                   # (this is desirable)
    max_run_length=MAX_RUN_LENGTH, # maximum length for a stable period
    min_odd_change=MIN_ODD_CHANGE, # minimum change in odd at a change point 
    apply_min_for_last_run_length=False,
    ):
    """
    This function generates a "jumping" probability sequence of length n_trials.
    The probability of a jump in the probability at any trial is p_c.
    """
    L = n_trials
    LJumpMax = max_run_length # maximum length for a stable period
    MinOddChange = min_odd_change # fold change (from pre to post jump) of odd for at least 1 transition
    pMin = p1_min # minimum value for probabilities
    pMax = p1_max # maximum value for probabilities
    """
    Define jumps occurrence with a geometric law of parameter p_c
    CDF = 1-(1-p_c)^k -> k = log(1-CDF)/log(1-p_c)
    The geometric distribution can be sampled uniformly from CDF.
    """
    SubL = []
    while sum(SubL) < L:
        if (apply_min_for_last_run_length and ((L - sum(SubL)) < min_run_length)):
            # restart from scratch as the last run length cannot be above the minimum
            SubL = [] 
        RandGeom = None
        while (RandGeom is None
            or RandGeom > max_run_length # note the change from >= to > from the original code
            or RandGeom < min_run_length):
            if p_c > 0:
                RandGeom = round(np.log(1-np.random.rand()) / np.log(1-p_c))
            else:
                assert L <= max_run_length
                RandGeom = L
        SubL.append(RandGeom)

    # Define probabilities
    tmp_p1 = np.zeros(len(SubL))

    for kJump in range(len(SubL)):
        if kJump == 0:
            tmp_p1[0] = np.random.rand()*(pMax-pMin) + pMin
        else:
            while True:
                tmp_p1[kJump] = np.random.rand()*(pMax-pMin) + pMin

                # compute transition odd change from pre to post jump
                oddChange = ((tmp_p1[kJump-1])/(1-tmp_p1[kJump-1])) /\
                    ((tmp_p1[kJump])/(1-tmp_p1[kJump]))

                # take this value if the odd change is sufficiently large
                if abs(np.log(oddChange)) > np.log(min_odd_change):
                    break

    # assign probs value to each trial
    p1 = np.zeros(L)
    p1[0:int(SubL[0])] = tmp_p1[0]
    for kJump in range(1, len(SubL)):
        p1[int(sum(SubL[0:int(kJump-1)])):int(sum(SubL[0:kJump]))] = tmp_p1[kJump-1] # this was kJump in the original code
    p1[int(sum(SubL[0:-1])):] = tmp_p1[-1]

    change_trials = np.cumsum(SubL[:-1], dtype=int)

    return p1, change_trials

def generate_gaussian_means_sequence(n_trials,
    p_c=GAUSSIAN_P_C,
    mean_min=GAUSSIAN_MEAN_MIN,
    mean_max=GAUSSIAN_MEAN_MAX,
    min_run_length=GAUSSIAN_MIN_RUN_LENGTH,
    max_run_length=None,
    min_odd_change=GAUSSIAN_MIN_ODD_CHANGE,
    apply_min_for_last_run_length=False
    ):
    if max_run_length is None:
        max_run_length = n_trials
    return generate_p1s_sequence(n_trials, p_c=p_c,
        p1_min=mean_min,
        p1_max=mean_max,
        min_run_length=min_run_length,
        max_run_length=max_run_length,
        min_odd_change=min_odd_change,
        apply_min_for_last_run_length=apply_min_for_last_run_length)

def random_outcomes_from_p1s(p1s):
    """
    Return a random sequence of binary outcomes using the given sequence of probabilities.
    Each value is sampled from a Bernoulli distribution. The probability represents
    the probability of this value being equal to 1.
    The input and output sequences are numpy array.
    """
    runif = np.random.random_sample(p1s.shape)
    seq = np.where((runif > p1s), 0, 1).astype(int)
    return seq

def random_gaussian_outcomes_from_means(means,
    std=GAUSSIAN_STD,
    min=GAUSSIAN_OUTCOME_MIN, max=GAUSSIAN_OUTCOME_MAX):
    """
    Return a random sequence of continuous outcomes sampled from a Gaussian distribution
    using the given sequence of means and a fixed standard deviation.
    The sampled values are clipped between the given min and max if provided.
    the probability of this value being equal to 1.
    The input and output sequences are numpy array.
    """
    outcomes = means + np.random.normal(scale=std, size=means.shape)
    outcomes = np.clip(outcomes, min, max)
    return outcomes

def set_seed(seed):
    """This function sets a random seed for a given seed number."""
    np.random.seed(seed)
    rd.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", type=str)
    parser.add_argument("-v", "--verbose", action='store_true', default=False)
    # for commands 'generate_ada-prob', 'generate_ada-pos' and 'combine'
    parser.add_argument("-o-json", "--output_json", type=str, default=None)
    parser.add_argument("-o-csv", "--output_csv", type=str, default=None)
    parser.add_argument("-o-js", "--output_js", type=str, default=None)
    parser.add_argument("-o-js-varname", "--output_js_variable_name", type=str,
        default="sequenceData")
    # for commands 'generate_ada-prob' and 'generate_ada-pos'
    parser.add_argument("--n_sessions", type=int, default=N_SESSIONS)
    parser.add_argument("--n_trials", type=int, default=N_TRIALS)
    parser.add_argument("--p_c", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min_odd_change", type=float, default=None)
    parser.add_argument("--min_run_length", type=int, default=None)
    parser.add_argument("--max_run_length", type=int, default=None)
    parser.add_argument("--apply_min_for_last_run_length", action='store_true', default=False)
    # for command 'generate_ada-pos'
    parser.add_argument("--gaussian_std", type=float, default=None)
    parser.add_argument("--gaussian_mean_min", type=float, default=None)
    parser.add_argument("--gaussian_mean_max", type=float, default=None)
    parser.add_argument("--gaussian_outcome_min", type=float, default=None)
    parser.add_argument("--gaussian_outcome_max", type=float, default=None)
    # for command 'combine'
    parser.add_argument("--task_names", type=str, nargs='+')
    parser.add_argument("-i-jsons", "--input_jsons", type=str, nargs='+')
    # for command 'show'
    parser.add_argument("-i-json", "--input_json", type=str)
    args = parser.parse_args()

    cmd = args.command
    if cmd == "generate_ada-prob":
        task_name = "ada-prob"
        n_sessions = args.n_sessions
        n_trials = args.n_trials
        p_c = (args.p_c if args.p_c is not None
            else P_C)
        min_odd_change = (args.min_odd_change if args.min_odd_change is not None
            else MIN_ODD_CHANGE)
        min_run_length = (args.min_run_length if args.min_run_length is not None
            else MIN_RUN_LENGTH)
        max_run_length = (args.max_run_length if args.max_run_length is not None
            else MAX_RUN_LENGTH)
        apply_min_for_last_run_length = args.apply_min_for_last_run_length
        verbose = args.verbose
        if args.seed:
            set_seed(args.seed)

        p1s = np.zeros((n_sessions, n_trials))
        did_p1_change = np.zeros_like(p1s, dtype=bool)
        did_p1_change[:, 0] = True
        n_change_points = 0
        for i_sess in range(n_sessions):
            p1s_sess, change_trials = generate_p1s_sequence(n_trials, p_c,
                min_odd_change=min_odd_change, min_run_length=min_run_length,
                max_run_length=max_run_length,
                apply_min_for_last_run_length=apply_min_for_last_run_length)
            p1s[i_sess] = p1s_sess
            did_p1_change[i_sess, change_trials] = True
            n_change_points += len(change_trials)
        outcomes = random_outcomes_from_p1s(p1s)
        io_estimates = model.ideal_learner_prob_estimates_from_outcomes(outcomes, p_c,
            p1_min=P1_MIN, p1_max=P1_MAX)

        if verbose:
            print("p1s", p1s)
            print("outcomes", outcomes)
            print("io_estimates", io_estimates)
            print("total num. of change points", n_change_points)
            print("avg num. of change points", n_change_points / n_sessions)

        session_indices = np.repeat(np.arange(n_sessions)[:, np.newaxis], n_trials, axis=1)
        trial_indices = np.repeat(np.arange(n_trials)[np.newaxis, :], n_sessions, axis=0)
        data_dict = {
            "taskName": task_name,
            "nSessions": n_sessions,
            "nTrials": n_trials,
            "pC": p_c,
            "p1Min": P1_MIN,
            "p1Max": P1_MAX,
            "minOddChange": min_odd_change,
            "minRunLength": min_run_length,
            "maxRunLength": max_run_length,
            "applyMinForLastRunLength": apply_min_for_last_run_length,
            "sessionIndex": session_indices,
            "trialIndex": trial_indices,
            "p1": p1s,
            "didP1Change": did_p1_change,
            "outcome": outcomes,
            "idealObserverEstimate": io_estimates,
        }

        if args.output_json or args.output_js:
            data_jsonserializable = {
                key: (value.tolist() if type(value) == np.ndarray
                    else value) for (key, value) in data_dict.items()
            }
            if args.output_json:
                with open(args.output_json, 'w') as out_json_file:
                    json.dump(data_jsonserializable, out_json_file, indent=2)
            if args.output_js:
                varname = args.output_js_variable_name
                txt = f"const {varname} = {json.dumps(data_jsonserializable, indent=2)};\n"
                with open(args.output_js, 'w') as out_js_file:
                    out_js_file.write(txt)

        if args.output_csv:
            data_dict_df = {
                key: (value.reshape(-1) if type(value) == np.ndarray
                    else value) for (key, value) in data_dict.items()
            }
            # print(data_dict_df)
            data_df = pd.DataFrame.from_dict(data_dict_df)
            data_df.to_csv(args.output_csv)

    elif cmd == "generate_ada-pos":
        task_name = "ada-pos"
        n_sessions = args.n_sessions
        n_trials = args.n_trials
        p_c = (args.p_c if args.p_c is not None
            else GAUSSIAN_P_C)
        min_odd_change = (args.min_odd_change if args.min_odd_change is not None
            else GAUSSIAN_MIN_ODD_CHANGE)
        min_run_length = (args.min_run_length if args.min_run_length is not None
            else GAUSSIAN_MIN_RUN_LENGTH)
        max_run_length = (args.max_run_length if args.max_run_length is not None
            else GAUSSIAN_MAX_RUN_LENGTH)
        apply_min_for_last_run_length = args.apply_min_for_last_run_length
        gaussian_std = (args.gaussian_std if args.gaussian_std is not None
            else GAUSSIAN_STD)
        gaussian_mean_min = (args.gaussian_mean_min if args.gaussian_mean_min is not None
            else GAUSSIAN_MEAN_MIN)
        gaussian_mean_max = (args.gaussian_mean_max if args.gaussian_mean_max is not None
            else GAUSSIAN_MEAN_MAX)
        gaussian_outcome_min = (args.gaussian_outcome_min if args.gaussian_outcome_min is not None
            else GAUSSIAN_OUTCOME_MIN)
        gaussian_outcome_max = (args.gaussian_outcome_max if args.gaussian_outcome_max is not None
            else GAUSSIAN_OUTCOME_MAX)
        verbose = args.verbose
        if args.seed:
            set_seed(args.seed)

        means = np.zeros((n_sessions, n_trials))
        did_mean_change = np.zeros_like(means, dtype=bool)
        did_mean_change[:, 0] = True
        n_change_points = 0
        for i_sess in range(n_sessions):
            means_sess, change_trials = generate_gaussian_means_sequence(n_trials,
                p_c=p_c,
                mean_min=gaussian_mean_min,
                mean_max=gaussian_mean_max,
                min_run_length=min_run_length,
                max_run_length=max_run_length,
                min_odd_change=min_odd_change,
                apply_min_for_last_run_length=apply_min_for_last_run_length,
                )
            means[i_sess] = means_sess
            did_mean_change[i_sess, change_trials] = True
            n_change_points += len(change_trials)
        outcomes = random_gaussian_outcomes_from_means(means,
                    std=gaussian_std,
                    min=gaussian_outcome_min, max=gaussian_outcome_max)
        # io_estimates = ideal_observer_estimates_from_outcomes(outcomes, p_c)

        if verbose:
            print("p1s", p1s)
            print("outcomes", outcomes)
            # print("io_estimates", io_estimates)
            print("total num. of change points", n_change_points)
            print("avg num. of change points", n_change_points / n_sessions)

        session_indices = np.repeat(np.arange(n_sessions)[:, np.newaxis], n_trials, axis=1)
        trial_indices = np.repeat(np.arange(n_trials)[np.newaxis, :], n_sessions, axis=0)
        data_dict = {
            "taskName": task_name,
            "nSessions": n_sessions,
            "nTrials": n_trials,
            "pC": p_c,
            "std": gaussian_std,
            "meanMin": gaussian_mean_min,
            "meanMax": gaussian_mean_max,
            "minOddChange": min_odd_change,
            "minRunLength": min_run_length,
            "maxRunLength": max_run_length,
            "applyMinForLastRunLength": apply_min_for_last_run_length,
            "sessionIndex": session_indices,
            "trialIndex": trial_indices,
            "mean": means,
            "didMeanChange": did_mean_change,
            "outcome": outcomes,
            # "idealObserverEstimate": io_estimates,
        }

        if args.output_json or args.output_js:
            data_jsonserializable = {
                key: (value.tolist() if type(value) == np.ndarray
                    else value) for (key, value) in data_dict.items()
            }
            if args.output_json:
                with open(args.output_json, 'w') as out_json_file:
                    json.dump(data_jsonserializable, out_json_file, indent=2)
            if args.output_js:
                varname = args.output_js_variable_name
                txt = f"const {varname} = {json.dumps(data_jsonserializable, indent=2)};\n"
                with open(args.output_js, 'w') as out_js_file:
                    out_js_file.write(txt)

        if args.output_csv:
            data_dict_df = {
                key: (value.reshape(-1) if type(value) == np.ndarray
                    else value) for (key, value) in data_dict.items()
            }
            # print(data_dict_df)
            data_df = pd.DataFrame.from_dict(data_dict_df)
            data_df.to_csv(args.output_csv)

    elif cmd == "combine":
        task_names = args.task_names
        input_jsons = args.input_jsons
        data_dict_by_task_name = {}
        for i, in_json in enumerate(input_jsons):
            with open(in_json, 'r') as in_file:
                data_dict_by_task_name[task_names[i]] = json.load(in_file)

        if args.output_json:
            with open(args.output_json, 'w') as out_json_file:
                json.dump(data_dict_by_task_name, out_json_file, indent=2)
        if args.output_js:
            varname = args.output_js_variable_name
            txt = f"const {varname} = {json.dumps(data_dict_by_task_name, indent=2)};\n"
            with open(args.output_js, 'w') as out_js_file:
                out_js_file.write(txt)
        if args.output_csv:
            dfs = []
            for task_name, data_dict in data_dict_by_task_name.items():
                # flatten
                data_dict_flat = {
                    key: np.array(v).reshape(-1) if type(v) == list else v
                    for (key, v) in data_dict.items()
                }
                df = pd.DataFrame.from_dict(data_dict_flat)
                if "taskName" not in df:
                    df["taskName"] = task_name
                dfs += [df]
            df = pd.concat(dfs)
            df.to_csv(args.output_csv)


    elif cmd == "show":
        input_json = args.input_json
        with open(args.input_json, 'r') as in_json_file:
            session_data = json.load(in_json_file)
        estimates = np.array(session_data["estimate"])
        io_estimates = np.array(session_data["idealObserverEstimate"])
        outcomes = np.array(session_data["outcome"], dtype=float)
        p1s = np.array(session_data["p1"])
        mse = np.mean((estimates - p1s) ** 2)
        mse_io = np.mean((io_estimates - p1s) ** 2)
        score = (SCORE_MIN_MSE - mse) / (SCORE_MIN_MSE - mse_io)
        with np.printoptions(precision=2, suppress=True, floatmode='fixed'):
            print("outcomes", outcomes)
            print("p1s", p1s)
            print("estimate", estimates)
            print("io_estimate", io_estimates)
        print("mse", mse)
        print("io_mse", mse_io)
        print("score", score)

    elif cmd == "compute_io_example_prob":
        p_c = (args.p_c if args.p_c is not None
            else P_C)
        outcomes = np.array([1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1,
            0, 0, 0, 0, 0, 1, 0])
        io_estimates = model.ideal_learner_prob_estimates_from_outcomes(outcomes, p_c,
            p1_min=P1_MIN, p1_max=P1_MAX)
        print("example outcomes", outcomes)
        print("example IO estimates", np.around(io_estimates, decimals=2))

    elif cmd == "compute_example_pos":
        p_c = (args.p_c if args.p_c is not None
            else P_C)
        gaussian_std = (args.gaussian_std if args.gaussian_std is not None
            else GAUSSIAN_STD)
        outcomes = np.array([0.33859375, 0.40578125, 0.3821875, 0.29453125])
        estimates = model.reduced_learner_pos_estimates_from_outcomes(outcomes, p_c,
            std_gen=gaussian_std)
        oracle_estimates = np.cumsum(outcomes) / np.arange(1, len(outcomes) + 1)
        print("example outcomes", outcomes)
        print("example oracle estimates", np.around(oracle_estimates, decimals=4))
        print("example model estimates", np.around(estimates, decimals=4))

        
    elif cmd == "compute_mae":
        n_sessions = args.n_sessions
        n_trials = args.n_trials
        p_c = (args.p_c if args.p_c is not None
            else P_C)
        min_odd_change = (args.min_odd_change if args.min_odd_change is not None
            else MIN_ODD_CHANGE)
        min_run_length = (args.min_run_length if args.min_run_length is not None
            else MIN_RUN_LENGTH)
        max_run_length = (args.max_run_length if args.max_run_length is not None
            else MAX_RUN_LENGTH)
        apply_min_for_last_run_length = args.apply_min_for_last_run_length
        verbose = args.verbose
        if args.seed:
            set_seed(args.seed)

        p1s = np.zeros((n_sessions, n_trials))
        did_p1_change = np.zeros_like(p1s, dtype=bool)
        did_p1_change[:, 0] = True
        n_change_points = 0
        for i_sess in range(n_sessions):
            p1s_sess, change_trials = generate_p1s_sequence(n_trials, p_c,
                min_odd_change=min_odd_change, min_run_length=min_run_length,
                max_run_length=max_run_length,
                apply_min_for_last_run_length=apply_min_for_last_run_length)
            p1s[i_sess] = p1s_sess
            did_p1_change[i_sess, change_trials] = True
            n_change_points += len(change_trials)
        outcomes = random_outcomes_from_p1s(p1s)
        io_estimates = model.ideal_learner_prob_estimates_from_outcomes(outcomes, p_c,
            p1_min=P1_MIN, p1_max=P1_MAX)

        io_mae_per_sess = np.mean(np.abs(io_estimates - p1s), axis=1)
        still_mae_per_sess = np.mean(np.abs(0.5 - p1s), axis=1)
        print("mean absolute error per sess stay still: ", still_mae_per_sess)
        print("mean absolute error per sess ideal observer: ", io_mae_per_sess)

        io_mae = np.mean(np.abs(io_estimates - p1s))
        still_mae = np.mean(np.abs(0.5 - p1s))

        print("mean absolute error stay still: ", still_mae)
        print("mean absolute error ideal observer: ", io_mae)
