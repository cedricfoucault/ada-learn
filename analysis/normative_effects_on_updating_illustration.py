"""
Plot an illustration of the normative posterior updating and the effects
of change-point probability and uncertainty in this updating process.
Illustrate those effects in each of the two learning tasks.
"""

import data_analysis_utils as dana
import matplotlib as mpl
import matplotlib.pyplot as plt
import model_learner as model
import numpy as np
import plot_utils as plut
import scipy.stats

FNAME_PREFIX = "normative_effects_on_updating_illustration"

FULLWIDTH_MARGIN = 0.6
FIGWIDTH = (plut.A4_PAPER_CONTENT_WIDTH - FULLWIDTH_MARGIN) / 4
FIGSIZE = (FIGWIDTH, FIGWIDTH * 2/3)
COLOR_BY_LEVEL = ["#2aa02b", "#e378c1"]
ALPHA_OVERLAP = 0.5
HATCH_OVERLAP = "//"

output_dir = dana.get_output_dir_for_group_def_file(dana.ADA_LEARN_GROUP_DEF_FILE,
        create_if_needed=True)

def plot_fun(ax, x, fx, show_mean=False, color=plut.BLACK_COLOR,
        alpha=None, zorder=None,
        xformat="{:.1f}".format,
        yformat="{:.0f}".format):
    ax.plot(x, fx, '-', color=color, alpha=alpha, zorder=zorder)
    if show_mean:
        mean = np.sum(x * fx) / np.sum(fx)
        ax.axvline(x=mean, ls='--', color=color, alpha=alpha, lw=1.)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(xformat))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(yformat))

def normalized_pdf(unnorm_dist, dx):
    return unnorm_dist / (np.sum(unnorm_dist) * dx)

def fill_with_hatch(ax, x, y, color=None, alpha=ALPHA_OVERLAP, hatch=HATCH_OVERLAP):
    # workaround a bug in matplotlib:
    # when rendering in PDF, the hatches don't show if they have been drawn with alpha
    ax.fill_between(x, y, color=color, alpha=alpha)
    ax.fill_between(x, y, color="None", edgecolor=color, hatch=HATCH_OVERLAP)


def plot_fig_prior(xvals, dx, prior, color=plut.BLACK_COLOR,
    yformat="{:.0f}".format):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.gca()
    plot_fun(ax, xvals, prior, show_mean=True, color=color, yformat=yformat)
    return fig

def plot_fig_likelihood_without_overlap(xvals, dx, prior, lik, color,
    yformat="{:.0f}".format):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.gca()
    plot_fun(ax, xvals, lik, show_mean=False, color=plut.BLACK_COLOR,
        yformat=yformat)
    return fig

def plot_fig_likelihood_with_overlap_prior(xvals, dx, prior, lik, color,
    yformat="{:.0f}".format):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.gca()
    plot_fun(ax, xvals, lik, show_mean=False, color=color,
        yformat=yformat)
    # overlay prior
    plot_fun(ax, xvals, prior,
        show_mean=False, color=plut.BLACK_COLOR, alpha=0.2, zorder=-1,
        yformat=yformat)
    # show overlap
    fill_with_hatch(ax, xvals, np.fmin(prior, lik), color=color)
    return fig

def plot_fig_posterior(xvals, dx, prior, lik, color,
    yformat="{:.0f}".format):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.gca()
    # posterior = product of the prior and likelihood, normalized
    plot_fun(ax, xvals, normalized_pdf((prior * lik), dx),
            show_mean=True, color=color,
            yformat=yformat)
    # overlay prior
    plot_fun(ax, xvals, prior,
        show_mean=True, color=plut.BLACK_COLOR, alpha=0.2, zorder=-1,
            yformat=yformat)
    return fig

def equalize_ylim(figures):
    nrows = len(figures)
    ncols = len(figures[0])
    ymin, ymax = figures[0][0].gca().get_ylim()
    for row in range(nrows):
        for col in range(ncols):
            fig = figures[row][col]
            ylim = fig.gca().get_ylim()
            ymin = min(ymin, ylim[0])
            ymax = max(ymax, ylim[1])
    for row in range(nrows):
        for col in range(ncols):
            fig = figures[row][col]
            fig.gca().set_ylim(ymin, ymax)

def save_figures_with_names(figures, fignames):
    nrows = len(figures)
    ncols = len(figures[0])
    for row in range(nrows):
        for col in range(ncols):
            fig = figures[row][col]
            figname = fignames[row][col]
            for ext in ['png', 'pdf']:
                figpath = dana.get_path(output_dir, figname, ext=ext)
                plut.save_figure(fig, figpath)

def run():
    plut.setup_mpl_style()

    #
    # Illustration of the two determinants themselves (not their effects on the update)
    #
    resol = 1001
    xvals = np.linspace(start=0., stop=1., num=resol)
    dx = xvals[1] - xvals[0]
    prior_mean = 1/3
    prior_scale = 1/15
    lik_mean = 0.55
    lik_scale = 1/8
    prior = (0.75 * scipy.stats.norm(loc=prior_mean, scale=prior_scale).pdf(xvals)
        + 0.25 * scipy.stats.uniform(0, 1).pdf(xvals))
    prior = normalized_pdf(prior, dx)

    lik = scipy.stats.norm(loc=lik_mean, scale=lik_scale).pdf(xvals)
    lik = normalized_pdf(lik, dx)

    fig = plt.figure(figsize=(FIGWIDTH, FIGWIDTH / 2))
    ax = fig.gca()
    plot_fun(ax, xvals, prior, show_mean=False, color=plut.BLACK_COLOR)
    plot_fun(ax, xvals, lik, show_mean=False, color=plut.BLACK_COLOR)
    # show overlap
    fill_with_hatch(ax, xvals, np.fmin(prior, lik), color=plut.BLACK_COLOR)
    ax.set_ylim(0, None)
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_visible(False)
    figname = f"{FNAME_PREFIX}_determinants"
    for ext in ['png', 'pdf']:
        figpath = dana.get_path(output_dir, figname, ext=ext)
        plut.save_figure(fig, figpath)

    nrows = 3
    ncols = 2
    figures = [[None for col in range(ncols)] for row in range(nrows)]
    fignames = [[None for col in range(ncols)] for row in range(nrows)]

    #
    # Illustration of the effect of change-point probability in the magnitude learning task
    #

    std_gen = 10/300
    mean_gen = 0.33
    run_length = 4

    seed = 1
    np.random.seed(seed)
    outcomes = scipy.stats.norm.rvs(loc=mean_gen, scale=std_gen, size=run_length)
    p_c = 1/10
    inf_dict = model.reduced_learner_pos_inference_from_outcomes(
                        outcomes, p_c=p_c, std_gen=std_gen)
    confs = model.reduced_learner_pos_confidence_with_type(
        inf_dict, conf_type="prev-conf-pred", std_gen=std_gen)
    prior_sds = dana.uncertainty_from_confidence(confs)
    prior_sd = prior_sds[-1]
    prior_mean = inf_dict["mean"][-1]

    resol = 1001
    xvals = np.linspace(start=0., stop=1., num=resol)
    dx = xvals[1] - xvals[0]
    prior = ((1-p_c) * scipy.stats.norm(loc=prior_mean, scale=prior_sd).pdf(xvals)
        + p_c * scipy.stats.uniform(0, 1).pdf(xvals))
    prior = normalized_pdf(prior, dx)

    next_outcomes = [0.37, 0.66]
    # On different figures, plot, for each level of the variable, the prior,
    # the likelihood, and the posterior resulting from the multiplication of the two
    for i_level in range(2):
        color = COLOR_BY_LEVEL[i_level]
        basename = f"{FNAME_PREFIX}_cpp-ada-pos"
        # Plot prior
        fig = plot_fig_prior(xvals, dx, prior)
        fignames[0][i_level] = f"{basename}_level-{i_level+1}_prior"
        figures[0][i_level] = fig
        # Plot likelihood function
        lik = normalized_pdf(
            scipy.stats.norm(loc=next_outcomes[i_level], scale=std_gen).pdf(xvals), dx)
        fig = plot_fig_likelihood_with_overlap_prior(xvals, dx, prior, lik, color)
        fignames[1][i_level] = f"{basename}_level-{i_level+1}_likelihood"
        figures[1][i_level] = fig
        # Plot posterior = product of the two, normalized
        fig = plot_fig_posterior(xvals, dx, prior, lik, color)
        fignames[2][i_level] = f"{basename}_level-{i_level+1}_posterior"
        figures[2][i_level] = fig

    # Equalize ylim across figures
    equalize_ylim(figures)
    # Save the figures
    save_figures_with_names(figures, fignames)

    #
    # Illustration of the effect of uncertainty in the magnitude learning task
    #

    std_gen = 10/300
    mean_gen = 0.33
    run_length = 10

    seed = 1
    np.random.seed(seed)
    outcomes = scipy.stats.norm.rvs(loc=mean_gen, scale=std_gen, size=run_length)
    p_c = 1/10
    inf_dict = model.reduced_learner_pos_inference_from_outcomes(
                        outcomes, p_c=p_c, std_gen=std_gen)
    confs = model.reduced_learner_pos_confidence_with_type(
        inf_dict, conf_type="prev-conf-pred", std_gen=std_gen)
    prior_sds = dana.uncertainty_from_confidence(confs)
    prior_mean = inf_dict["mean"][-1]

    resol = 1001
    xvals = np.linspace(start=0., stop=1., num=resol)
    dx = xvals[1] - xvals[0]

    next_outcome = 0.4
    prior_sds = prior_sds[-1], prior_sds[0]
    for i_level in range(2):
        color = COLOR_BY_LEVEL[i_level]
        basename = f"{FNAME_PREFIX}_uncertainty-ada-pos"
        # Plot prior
        prior = ((1-p_c) * scipy.stats.norm(loc=prior_mean, scale=prior_sds[i_level]).pdf(xvals)
                + p_c * scipy.stats.uniform(0, 1).pdf(xvals))
        prior = normalized_pdf(prior, dx)
        fig = plot_fig_prior(xvals, dx, prior, color=color)
        fignames[0][i_level] = f"{basename}_level-{i_level+1}_prior"
        figures[0][i_level] = fig
        # Plot likelihood function
        lik = normalized_pdf(
            scipy.stats.norm(loc=next_outcome, scale=std_gen).pdf(xvals), dx)
        fig = plot_fig_likelihood_without_overlap(xvals, dx, prior, lik, color)
        fignames[1][i_level] = f"{basename}_level-{i_level+1}_likelihood"
        figures[1][i_level] = fig
        # Plot posterior = product of the two, normalized
        fig = plot_fig_posterior(xvals, dx, prior, lik, color)
        fignames[2][i_level] = f"{basename}_level-{i_level+1}_posterior"
        figures[2][i_level] = fig

    # Equalize ylim across figures
    equalize_ylim(figures)
    # Save the figures
    save_figures_with_names(figures, fignames)

    #
    # Illustration of the effect of change-point probability in the probability learning task
    #

    basename = f"{FNAME_PREFIX}_cpp-ada-prob"

    # Compute prior
    seed = 1
    # n_outcomes = 4
    # outcomes = np.zeros((n_outcomes, ))
    outcomes = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    p_c = 1/20
    resol = 101
    pvals = np.linspace(start=0., stop=1., num=resol)
    dx = pvals[1] - pvals[0]
    inf_dict = model.ideal_learner_prob_inference_from_outcomes(outcomes, p_c,
        resol=resol,
        do_inference_on_current_trial=False)
    dist = inf_dict["dist"]
    mean = inf_dict["mean"]
    filt = (pvals >= model.P1_MIN) & (pvals <= model.P1_MAX)
    pvals = pvals[filt]
    dist = dist[:, filt]
    prior = dist[-1, :]
    prior = normalized_pdf(prior, dx)

    next_outcomes = [0, 1]
    for i_level in range(2):
        color = COLOR_BY_LEVEL[i_level]
        # Plot prior
        fig = plot_fig_prior(pvals, dx, prior, yformat="{:.1f}".format)
        fignames[0][i_level] = f"{basename}_level-{i_level+1}_prior"
        figures[0][i_level] = fig
        # Plot likelihood function
        next_outcome = next_outcomes[i_level]
        lik = pvals if next_outcome == 1 else (1 - pvals)
        fig = plot_fig_likelihood_with_overlap_prior(pvals, dx, prior, lik, color,
            yformat="{:.1f}".format)
        fignames[1][i_level] = f"{basename}_level-{i_level+1}_likelihood"
        figures[1][i_level] = fig
        # Plot posterior = product of the two, normalized
        fig = plot_fig_posterior(pvals, dx, prior, lik, color,
            yformat="{:.1f}".format)
        fignames[2][i_level] = f"{basename}_level-{i_level+1}_posterior"
        figures[2][i_level] = fig

    # Equalize ylim across figures
    equalize_ylim(figures)
    # Save the figures
    save_figures_with_names(figures, fignames)

    #
    # Illustration of the effect of uncertainty in the probability learning task
    #

    basename = f"{FNAME_PREFIX}_uncertainty-ada-prob"

    do_cps = [False, True]
    seeds = [10, 0]
    times = [41, 42]

    for i_level in range(2):
        seed = seeds[i_level]
        t = times[i_level]
        do_cp = do_cps[i_level]
        np.random.seed(seed)
        # Generate outcome sequence
        run_length = 40
        if do_cp:
            p1_1 = 0.3
            p1_2 = 0.9
            p1s = np.array([p1_1] * run_length + [p1_2] * run_length)
        else:
            # p1_1 = 0.55
            p1_1 = 0.65
            p1s = np.array([p1_1] * 2 * run_length)
        runif = np.random.random_sample(p1s.shape)
        outcomes = np.where((runif > p1s), 0, 1).astype(int)
        # Compute priors
        p_c = 1/20
        resol = 101
        pvals = np.linspace(start=0., stop=1., num=resol)
        dx = pvals[1] - pvals[0]
        inf_dict = model.ideal_learner_prob_inference_from_outcomes(outcomes, p_c,
            resol=resol,
            do_inference_on_current_trial=False)
        dist = inf_dict["dist"]
        mean = inf_dict["mean"]
        filt = (pvals >= model.P1_MIN) & (pvals <= model.P1_MAX)
        pvals = pvals[filt]
        dist = dist[:, filt]
        color = COLOR_BY_LEVEL[i_level]
        # Plot prior, likelihood function and posterior for the selected time step
        prior = dist[t, :]
        prior = normalized_pdf(prior, dx)
        lik = pvals if outcomes[t+1] == 1 else (1 - pvals)
        mean_prev = mean[t]
        mean_next = mean[t+1]
        # Plot prior
        fig = plot_fig_prior(pvals, dx, prior, color=color, yformat="{:.1f}".format)
        fignames[0][i_level] = f"{basename}_level-{i_level+1}_prior"
        figures[0][i_level] = fig
        # Plot likelihood function
        fig = plot_fig_likelihood_without_overlap(pvals, dx, prior, lik, color,
            yformat="{:.1f}".format)
        fignames[1][i_level] = f"{basename}_level-{i_level+1}_likelihood"
        figures[1][i_level] = fig
        # Plot posterior = product of the two, normalized
        fig = plot_fig_posterior(pvals, dx, prior, lik, color,
            yformat="{:.1f}".format)
        fignames[2][i_level] = f"{basename}_level-{i_level+1}_posterior"
        figures[2][i_level] = fig

    # Equalize ylim across figures
    equalize_ylim(figures)
    # Save the figures
    save_figures_with_names(figures, fignames)

if __name__ == '__main__':
    run()
