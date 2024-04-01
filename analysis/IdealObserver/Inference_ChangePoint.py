#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted from https://github.com/florentmeyniel/TransitionProbModel
"""
import itertools
from operator import mul
from functools import reduce
from scipy.stats import dirichlet
import numpy as np


def get_clipped_grid_indices(Dir_grid, p1_min=None, p1_max=None):
    if (p1_min is not None) or (p1_max is not None):
        # clip each grid location where  p1 is not within [min, max]
        indices = []
        for i, (p0, p1) in enumerate(Dir_grid):
            if (((p1_min is not None) and (p1 < p1_min))
                or ((p1_max is not None) and (p1 > p1_max))):
                indices += [i]
        return indices
    else:
        return None


def likelihood_table(Nitem=2, resol=None):
    """
    Compute the likelihood of observations on a discretized parameter grid,
    for *Nitem* with *resol* values for each parameter.

    The observation likelihood is determined by Dirichlet
    parameters (or bernoulli parameters when *Nitem*=1); those parameters (and
    their combinations) are discretized using a grid. The number of dimensions
    of the grid is *Nitem*. The function outputs *Dir_grid*, which is a list
    of tuples, each tuple being a possible combination of dirichlet parameter
    values.

    The function outputs *observation_lik*, the likelihood of observations
    presented as a dictonary. The keys of this dictionary are the possible
    sequences of trailing and leading observations of interest given the
    specified order, and its values correspond to the discretized distribution
    of likelihoods into states. For instance, the key (0,1,2) corresponds to
    the sequence 0, then 1, then 2.

    """

    # Compute discretized parameter grid
    grid_param = np.linspace(0, 1, resol)

    # Get combinations of all possible (discretized) Dirichlet parameters
    # satisfying that their sum is 1.
    Dir_grid = [ntuple
                for ntuple in itertools.product(grid_param, repeat=Nitem)
                if np.isclose(sum(ntuple), 1)]

    # Compute likelihood of observation
    observation_lik = {}
    for item in range(Nitem):
        observation_lik[(item,)] = \
            np.array([combi[item] for combi in Dir_grid], dtype='float')

    return observation_lik, Dir_grid


def change_marginalize(curr_dist, clipped_grid_indices=None):
    """
    Compute the integral:
        \int p(\theta_t|y)p(\theta_{t+1}|\theta_t)) d\theta_t
        in which the transition matrix has zeros on the diagonal, and
        1/(n_state-1) elsewhere. In other words, it computes the updated
        distribution in the case of change point (but does not multiply by
        the prior probability of change point).

        NB: currently, the prior on transition is flat, and the prior on the
        base rate of occurence of item is also flat; we may want to change this
        latter aspect at least.

        Cedric Foucault: Change to handle cases where we forbid transitioning
        to states outside of the [p_min, p_max] clipping (corresponding to
        places where curr_dist has 0 probability)
    """
    if clipped_grid_indices is not None:
        n_unclipped = curr_dist.shape[0] - len(clipped_grid_indices)
        retval = (sum(curr_dist) - curr_dist) / (n_unclipped-1)
        retval[clipped_grid_indices] = 0
        return retval
    else:
        return (sum(curr_dist) - curr_dist) / (curr_dist.shape[0]-1)


def init_Alpha(Dir_grid=None, Dirichlet_param=None, clipped_grid_indices=None):
    """
    Initialize Alpha, which is the joint probability distribution of
    observations and parameter values.
    This initialization takes into account a bias in the dirchlet parameter
    (wit the constraint that the same bias applies to all transitions).
    Discretized state are sorted as in likelihood_table, such as the output
    of both functions can be combined.
    """
    # get discretized dirichlet distribution at quantitles' location
    dir_dist = np.array([dirichlet.pdf(np.array(grid), Dirichlet_param)
                for grid in Dir_grid])

    # set the probability to 0 at each clipped grid location
    if clipped_grid_indices is not None:
        dir_dist[clipped_grid_indices] = 0

    # normalize to a probability distribution
    dir_dist = dir_dist / sum(dir_dist)

    Alpha0 = dir_dist

    return np.array(Alpha0)


def forward_updating(seq=None, lik=None, p_c=None, Alpha0=None,
    clipped_grid_indices=None):
    """
    Update iteratively the Alpha, the joint probability of observations and parameters
    values, moving forward in the sequence.
    Alpha[t] is the estimate given previous observation, the t-th included.
    """

    # Initialize containers
    Alpha = np.ndarray((len(Alpha0), len(seq)))
    Alpha_no_change = np.ndarray((len(Alpha0), len(seq)))
    Alpha_change = np.ndarray((len(Alpha0), len(seq)))

    # Compute iteratively
    for t in range(len(seq)):
        if t == 0:
            # Update Alpha with the new observation
            Alpha_no_change[:, t] = (1-p_c) * lik[tuple(seq[t:t+1])] * Alpha0
            Alpha_change[:, t] = p_c * lik[tuple(seq[t:t+1])] * change_marginalize(Alpha0,
                clipped_grid_indices=clipped_grid_indices)
            Alpha[:, t] = Alpha_no_change[:, t] + Alpha_change[:, t]

            # Normalize
            cst = sum(Alpha[:, t])
            Alpha_no_change[:, t] = Alpha_no_change[:, t]/cst
            Alpha_change[:, t] = Alpha_change[:, t]/cst
            Alpha[:, t] = Alpha[:, t]/cst
        else:
            # Update Alpha with the new observation
            Alpha_no_change[:, t] = (1-p_c) * lik[tuple(seq[t:t+1])] * Alpha[:, t-1]
            Alpha_change[:, t] = p_c * lik[tuple(seq[t:t+1])] *\
                change_marginalize(Alpha[:, t-1],
                    clipped_grid_indices=clipped_grid_indices)
            Alpha[:, t] = Alpha_no_change[:, t] + Alpha_change[:, t]

            # Normalize
            cst = sum(Alpha[:, t])
            Alpha_no_change[:, t] = Alpha_no_change[:, t]/cst
            Alpha_change[:, t] = Alpha_change[:, t]/cst
            Alpha[:, t] = Alpha[:, t]/cst

    return Alpha


def turn_posterior_into_prediction(Alpha=None, p_c=None, clipped_grid_indices=None):
    """
    Turn the posterior into a prediction, taking into account the possibility
    of a change point
    """
    # Check dimensions
    if len(Alpha.shape) == 1:
        Alpha = Alpha[:, np.newaxis]

    # Initialize containers
    pred_Alpha = np.ndarray(Alpha.shape)

    # Update
    for t in range(Alpha.shape[1]):
        # Update Alpha, without a new observation but taking into account
        # the possibility of a change point
        pred_Alpha[:, t] = (1-p_c) * Alpha[:, t] + \
                           p_c * change_marginalize(Alpha[:, t],
                                    clipped_grid_indices=clipped_grid_indices)

    return pred_Alpha


def marginal_Alpha(Alpha, lik):
    """
    Compute the marginal distributions for all Dirichlet parameters and
    transition types
    """
    marg_dist = {}
    for pattern in lik.keys():
        # get grid of values
        grid_val = np.unique(lik[pattern])

        # initialize container
        marg_dist[pattern] = np.zeros((len(grid_val), Alpha.shape[1]))

        # marginalize over the dimension corresponding to the other patterns
        for k, value in enumerate(grid_val):
            marg_dist[pattern][k, :] = np.sum(Alpha[(lik[pattern] == value), :], axis=0)

    return marg_dist


def compute_inference(seq=None, resol=None, Nitem=2, p_c=None,
    p1_min=None, p1_max=None, do_inference_on_current_trial=False):
    """
    Wrapper function that computes the posterior marginal distribution, starting
    from a sequence
    """
    lik, grid = likelihood_table(Nitem=Nitem, resol=resol)
    clipped_grid_indices = get_clipped_grid_indices(grid, p1_min=p1_min, p1_max=p1_max)
    Alpha0 = init_Alpha(Dir_grid=grid,
                        Dirichlet_param=[1 for k in range(Nitem)],
                        clipped_grid_indices=clipped_grid_indices)
    Alpha = forward_updating(seq=seq, lik=lik, p_c=p_c, Alpha0=Alpha0,
        clipped_grid_indices=clipped_grid_indices)
    if do_inference_on_current_trial:
        marg_Alpha = marginal_Alpha(Alpha, lik)
    else:
        pred_Alpha = turn_posterior_into_prediction(Alpha=Alpha, p_c=p_c,
            clipped_grid_indices=clipped_grid_indices)
        marg_Alpha = marginal_Alpha(pred_Alpha, lik)
    return marg_Alpha
