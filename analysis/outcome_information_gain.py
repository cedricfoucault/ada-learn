"""
Analyze the information gain afforded by a single outcome
in different studies previously conducted.

Information gain is measured by the KL divergence between the prior and
the posterior (having observed a single outcome),
using the prior as reference distirbution.

Note that this is a standard way of measuring information gain
(in particular when probability distributions are continuous).
See e.g.:
https://www.chemeurope.com/en/encyclopedia/Kullback%E2%80%93Leibler_divergence.html#KL_divergence_and_Bayesian_updating
    "In Bayesian statistics the KL divergence can be used as a measure of
    the information gain in moving from a prior distribution
    to a posterior distribution."
"""

import data_analysis_utils as dana
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plot_utils as plut
import os

FNAME_PREFIX = "outcome_information_gain"

MAGNITUDE_STUDIES_INFO = [
    {
        "name": "Nassar et al., 2012",
        "range": [0, 300],
        "likelihood_type": "gaussian",
        "sds": [10, 5]
    }, {
        "name": "McGuire et al., 2014",
        "range": [0, 300],
        "likelihood_type": "gaussian",
        "sds": [10, 25]
    }, {
        "name": "Vaghi et al., 2017",
        "range": [1, 360],
        "likelihood_type": "gaussian",
        "sds": [12]
    },
    {
        "name": "Prat-Carrabin et al., 2021",
        "range": [0, 300],
        "likelihood_type": "triangular",
        "half_widths": [20]
    },
    {
        "name": "Nassar et al., 2010",
        "range": [0, 300],
        "likelihood_type": "gaussian",
        "sds": [35, 25, 15, 5]
    }
]

# resolution of the grid used to discretize and
# approximate numerically continuous probability distributions
GRID_RESOL = 1001


BAR_COLOR = {
    "Magnitude learning": "#C0C0E4",
    "Probability learning": "#E4E4C0"
}

def get_value_grid(v_range, resol=GRID_RESOL):
    """
    Returns a 1D grid of evenly-spaced sample values over the possible range of values.

    Arguments:
    - v_range: The range (min, max) of the possible values.
    - resol: The resolution (number of points) of the grid
    """
    return np.linspace(v_range[0], v_range[1], num=resol)

def get_prior_prob_hidden(h_grid):
    """
    p(h)

    Returns the prior probability distribution about the hidden quantity,
    before observing any outcome, or after a change point.
    We take the prior to be a uniform distribution.

    Arguments:
    - h_grid: The discretized grid of possible values that the hidden quantity can take.
    """
    return np.ones_like(h_grid) / len(h_grid)


def likelihood_outcome_given_hidden_Bernoulli(x, h):
    """
    The likelihood function p(x | h), for the probability learning task.

    In that case h is a probability, and x | h is a Bernoulli distribution
    with Bernoulli parameter equal to h.
    """
    return x * h + (1 - x) * (1 - h)

def likelihood_outcome_given_hidden_Gaussian(x, h, sd):
    """
    The likelihood function p(x | h) for the magnitude learning task,
    when a Gaussian generative distribution is used (the most common case).

    In that case h is an average magnitude, and x | h is a Gaussian distribution
    with a mean equal to h and a given standard deviation, which is a task parameter.
    """
    return np.exp(-((x - h) / sd)**2 / 2) / np.sqrt(2 * np.pi * sd**2)

def likelihood_outcome_given_hidden_triangular(x, h, half_width):
    """
    The likelihood function p(x | h) for the magnitude learning task,
    when a Triangular generative distribution is used (only used in one study).

    Half_width parameterizes the variance of the distribution like the SD for the
    Gaussian case.
    """
    a = h - half_width
    b = h + half_width
    return np.where((x > a) & (x < b),
        (half_width - np.abs(h - x)) / (half_width ** 2), 0)


def get_posterior_prob_hidden_given_outcome(likelihood, prob_h_prior,
    h_axis=None):
    """
    p(h | x)

    Returns the posterior probability distribution about the hidden quantity h
    at the same trial after observing a single outcome x.

    If an array of outcomes are considered, this will return a 2D array where
    one axis corresponds to the different outcome values and the other axis
    corresponds to h, the support of the distribution ('h_axis').

    The dimensions of the numpy arrays that are given as arguments
    must be consistent (1d arrays indexing h only, or 2d arrays with e.g.
    dimension 1 corresponding to h and dimension 2 corresponding to x)

    Arguments:
    - likelihood: p(x_1 | h_1), the likelihood function, as a numpy array
    - prob_h_prior: the prior probability distribution about the hidden quantity
        before observing any outcome, as a numpy array
    - h_axis: the dimension in the array corresponding to the hidden quantity h.
    """
    # 1. Compute the unnormalized posterior p(x_1 | h_1) * p(h_1)
    post_unnorm = likelihood * prob_h_prior
    # 2. Normalize the posterior
    post = post_unnorm / np.sum(post_unnorm, axis=h_axis)

    return post


def likelihood_2d_outcome_given_hidden(x_grid, h_grid, likelihood_type,
    sd=None, half_width=None):
    """
    Returns p(x | h), as a 2D array,
    where the first dimension corresponds to possible values of h,
    and the second dimension corresponds to possible values of x.
    """
    x_grid_2d = x_grid[np.newaxis, :]
    h_grid_2d = h_grid[:, np.newaxis]

    if likelihood_type == "bernoulli":
        return likelihood_outcome_given_hidden_Bernoulli(x_grid_2d, h_grid_2d)
    elif likelihood_type == "gaussian":
        return likelihood_outcome_given_hidden_Gaussian(x_grid_2d, h_grid_2d, sd)
    elif likelihood_type == "triangular":
        return likelihood_outcome_given_hidden_triangular(x_grid_2d, h_grid_2d, half_width)

def compute_information_gain(x_grid, h_grid, likelihood_type, sd=None, half_width=None):
    """
    Returns the information gain provided by a single outcome x about the quantity h to be learned.,
    for each value of x given in the grid.
    """
    # Compute p(h)
    prob_h_prior = get_prior_prob_hidden(h_grid)
    # Compute p[h | x]
    likelihood_2d = likelihood_2d_outcome_given_hidden(x_grid, h_grid, likelihood_type,
        sd=sd, half_width=half_width)
    prob_h_prior_2d = prob_h_prior[:, np.newaxis]
    h_grid_2d = h_grid[:, np.newaxis]
    prob_h_posterior_2d = get_posterior_prob_hidden_given_outcome(
        likelihood_2d, prob_h_prior_2d, h_axis=0)
    # Compute information gain as KL divergence
    # D_KL[post || prior] == \sum[post(h) * log(post(h) / prior(h))].
    # To use bits as units of information gain, use the base 2 for the log.
    information_gain = np.sum(
        np.where(
            np.isclose(prob_h_posterior_2d, 0),
            0, # put 0s directly since computing 0 * log(0) produces nan values
            prob_h_posterior_2d * np.log2(prob_h_posterior_2d / prob_h_prior_2d))
        , axis=0)
    
    return information_gain

def run():
    # Plot the bar graph of information gain in probability learning vs
    # magnitude learning, considering all the task parameter values
    # that have been used in different previous magnitude learning studies.

    output_dir = dana.get_output_dir_for_group_def_file(dana.ADA_LEARN_GROUP_DEF_FILE,
        create_if_needed=True)

    # Compute and aggregate data for each bar of the bar graph
    bar_data = {k: [] for k in ["value", "label", "text", "color"]}
    # Add information gain bar for probability learning
    task = "Probability learning" 
    color = BAR_COLOR[task]
    x_grid = np.array([0, 1])
    h_grid = get_value_grid([0, 1])
    ig = compute_information_gain(x_grid, h_grid, "bernoulli").mean()
    bar_data["value"] += [ig]
    bar_data["label"] += [task]
    bar_data["text"] += [""]
    bar_data["color"] += [color]
    # Add information gain bar for each magnitude learning study
    task = "Magnitude learning"
    color = BAR_COLOR[task]
    did_add_task_label = False
    studies_info = MAGNITUDE_STUDIES_INFO
    for i_study, study in enumerate(studies_info):
        v_range = study["range"]
        x_grid = get_value_grid(v_range)
        h_grid = get_value_grid(v_range)
        if ((study["likelihood_type"] == "gaussian")
            and ("sds" in study)):
            for sd in study["sds"]:
                ig = compute_information_gain(x_grid, h_grid, "gaussian", sd=sd).mean()
                bar_data["value"] += [ig]
                if not did_add_task_label:
                    bar_data["label"] += [task]
                    did_add_task_label = True
                else:
                    bar_data["label"] += [""]
                bar_data["text"] += [study["name"]]
                bar_data["color"] += [color] 
        elif (study["likelihood_type"] == "triangular"): 
            for hw in study["half_widths"]:
                ig = compute_information_gain(x_grid, h_grid, "triangular", half_width=hw).mean()
                bar_data["value"] += [ig]
                if not did_add_task_label:
                    bar_data["label"] += [task]
                    did_add_task_label = True
                else:
                    bar_data["label"] += [""]
                bar_data["text"] += [study["name"]]
                bar_data["color"] += [color]
    # Plot the bar graph figure
    plut.setup_mpl_style()
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    n_bars = len(bar_data["value"])
    fig_height_per_bar = 0.5
    figsize = (plut.A4_PAPER_CONTENT_WIDTH * 3 / 4,
        fig_height_per_bar * n_bars)
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    bars = ax.barh(np.arange(n_bars), bar_data["value"],
        height=0.75,
        color=bar_data["color"],
        tick_label=bar_data["label"])
    xlim = ax.get_xlim()
    x_text = xlim[0] + 0.01 * (xlim[1] - xlim[0])
    for i_bar in range(n_bars):
        ax.text(x_text, i_bar, bar_data["text"][i_bar],
            ha="left", va="center", color="black")
    ax.set_xlabel("Information gain from one observation\n(KL-divergence[posterior||prior], bits)")
    ax.invert_yaxis() # top-to-bottom
    configure_bar_plot_axes(ax)
    figname = f"{FNAME_PREFIX}_studies"
    for ext in ['png', 'pdf']:
        figpath = dana.get_path(output_dir, figname, ext)
        plut.save_figure(fig, figpath)

def configure_bar_plot_axes(ax, show_minor=False, labelbottom=True):
    # Remove the spines and ticks and use grid lines instead
    ax.grid(True, axis='x', which='major', ls='--', lw=.5, c='k', alpha=.3)
    if show_minor:
        ax.grid(True, axis='x', which='minor', ls='--', lw=.5, c='k', alpha=.3)
    ax.tick_params(axis='both', which='both',
                   bottom=False, top=False, labelbottom=labelbottom,
                   left=False, right=False, labelleft=True)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

if __name__ == '__main__':
    run()
