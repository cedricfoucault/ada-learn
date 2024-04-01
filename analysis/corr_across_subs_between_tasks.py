"""
Plot the correlation across subjects of a behavioral measure between the two tasks.
"""
import argparse
import data_analysis_utils as dana
import data_behavioral_measures as bmeas
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import pingouin as pg
import plot_utils as plut
import scipy.stats

FNAME_PREFIX = "corr_across_subs_between_tasks"
BMEAS_KEYS_CORR_BETWEEN_TASKS = [
    "update-freq",
    "coef-z-uncertainty",
    "coef-z-cpp",
    ]

def run_with_args(group_def_file, do_scatterplot=False):
    group_def_file = args.group_def_file
    data_files = dana.get_subject_data_files_from_group_def(group_def_file)
    n_subjects = len(data_files)
    output_dir = dana.get_output_dir_for_group_def_file(group_def_file,
        create_if_needed=True)

    data = dana.load_data(data_files[0])
    n_tasks = data.get("nTasks", 1)
    assert n_tasks == 2
    tasks = [sessions[0]["taskName"] for sessions in data["taskSessionDataArrays"]]
    
    bmeas_vals_by_key = { k: np.empty((n_subjects, n_tasks))
        for k in BMEAS_KEYS_CORR_BETWEEN_TASKS}
    for i_sub, data_file in enumerate(data_files):
        data = dana.load_data(data_file)
        for i_task, task in enumerate(tasks):
            for task_sessions in data["taskSessionDataArrays"]:
                if task_sessions[0]["taskName"] == task:
                    sessions = task_sessions
                    break
            measures = bmeas.compute_behavioral_measures(sessions)
            for k in BMEAS_KEYS_CORR_BETWEEN_TASKS:
                bmeas_vals_by_key[k][i_sub, i_task] = measures[k]
    

    plut.setup_mpl_style()

    # Compute correlation of the different measures between the two tasks
    corrs_data = []
    for k in BMEAS_KEYS_CORR_BETWEEN_TASKS:
        use_partial_corr = ("update-freq" not in k)
        vals = bmeas_vals_by_key[k]
        if use_partial_corr:
            df = pd.DataFrame({
                'x1': vals[:, 0],
                'x2': vals[:, 1],
                'covar': np.mean(bmeas_vals_by_key['update-freq'], axis=1)
                })
            partial_corr_stats = pg.partial_corr(df, x='x1', y='x2', covar='covar')
            r = partial_corr_stats['r'].item()
            p = partial_corr_stats['p-val'].item()
        else:
            r, p = scipy.stats.pearsonr(vals[:, 0], vals[:, 1])
        corrs_data += [{'measure': k, 'r': r, 'p': p, 'is_partial_corr': use_partial_corr}]
        if do_scatterplot:
            # Plot a scatter plot of the individual measure between the two tasks
            figname = f"{FNAME_PREFIX}_{k}"
            text = f"r={r:.2f}"
            text += f" (p={p:.1e})"
            if use_partial_corr:
                text += " (partial corr.)"
            figname += ".png"
            fig = plt.figure(figsize=(plut.A4_PAPER_CONTENT_WIDTH / 2,
                                      plut.A4_PAPER_CONTENT_WIDTH / 2))
            ax = fig.gca()
            ax.set_xlabel(f"{tasks[0].capitalize()} {k}")
            ax.set_ylabel(f"{tasks[1].capitalize()} {k}")
            ax.plot(vals[:, 0], vals[:, 1], '.', ms=2, color=plut.GRAY_COLOR, alpha=0.8)
            ax.text(0.05, 0.95, text,
                ha="left", va="top", transform=ax.transAxes, fontsize=7)
            figpath = op.join(output_dir, figname)
            plut.save_figure(fig, figpath, do_fit_axis_labels=True)

    corrs_data = pd.DataFrame(corrs_data)
    corrs_data_fpath = dana.get_path(output_dir, FNAME_PREFIX, 'csv')
    plut.save_stats(corrs_data, corrs_data_fpath)
    corrs_data.to_csv(corrs_data_fpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group_def_file", type=str,
        default=dana.ADA_LEARN_GROUP_DEF_FILE)
    args = parser.parse_args()
    run_with_args(args.group_def_file)
