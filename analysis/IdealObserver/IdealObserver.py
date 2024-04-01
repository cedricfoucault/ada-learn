#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted from https://github.com/florentmeyniel/TransitionProbModel
"""

from . import Inference_ChangePoint as io_hmm
import numpy as np
from scipy.stats import dirichlet


def run_inference(seq, options=None):
    """
    run_inference is a wrapper that computes the posterior inference of generative
    probabilities of the sequence seq.
    - seq is the sequence (numpy array)
    - options is a dictionary
      'p_c': a priori volatility (for hmm)
      'resol': number of bins used for discretization (for hmm)
      'p1_min': the minimum value of the Bernoulli parameter (used to clip the Dirichlet prior distribution)
      'p1_max': the maximum value of the Bernoulli parameter (used to clip the Dirichlet prior distribution)
    """
    Nitem = 2
    options = check_options(options)

    resol, p_c, p1_min, p1_max, do_inference_on_current_trial = parse_options(options)

    # Get full posterior
    marg_post = io_hmm.compute_inference(seq=seq, resol=resol,
                                         Nitem=Nitem, p_c=p_c,
                                         p1_min=p1_min, p1_max=p1_max,
                                         do_inference_on_current_trial=do_inference_on_current_trial)

    # Fill output
    out = fill_output_hmm(marg_post, resol, Nitem)
    out['surprise'] = compute_surprise(seq, out)

    return out


def parse_options(options):
    """
    Parse options
    """
    if 'resol' in options.keys():
        resol = options['resol']
    else:
        resol = 10
    if 'p_c' in options.keys():
        p_c = options['p_c']
    else:
        raise ValueError('options should contain a key "p_c"')
    if 'p1_min' in options.keys():
        p1_min = options['p1_min']
    else:
        p1_min = None
    if 'p1_max' in options.keys():
        p1_max = options['p1_max']
    else:
        p1_max = None
    if 'do_inference_on_current_trial' in options.keys():
        # if this is true, do the inference on the current trial rather
        # than the next trial
        do_inference_on_current_trial = options['do_inference_on_current_trial']
    else:
        # default: do inference on the next trial
        do_inference_on_current_trial = False
    return resol, p_c, p1_min, p1_max, do_inference_on_current_trial


def check_options(options):
    checked_options = {}

    # use lower case for all options
    for item in options.keys():
        checked_options[item.lower()] = options[item]
    return checked_options


def compute_mean_of_dist(dist, pgrid):
    """ Compute mean of probability distribution"""
    return dist.transpose().dot(pgrid)


def compute_sd_of_dist(dist, pgrid, Nitem):
    """ Compute SD of probability distribution"""
    m = compute_mean_of_dist(dist, pgrid)
    v = dist.transpose().dot(pgrid**2) - m**2
    return np.sqrt(v)


def compute_surprise(seq, out):
    """
    Compute surprise.
    """
    surprise = np.nan * np.ones(len(seq))
    patterns = set(out.keys())
    for t in range(1, len(seq)):
        if (seq[t],) in patterns:
            surprise[t] = -np.log2(out[(seq[t],)]['mean'][t-1])
    return surprise


def fill_output_hmm(post, resol, Nitem):
    """
    Fill output strucutre of hmm inferece
    """
    out = {}
    pgrid = np.linspace(0, 1, resol)
    for item in post.keys():
        out[item] = {}
        out[item]['dist'] = post[item]
        out[item]['mean'] = compute_mean_of_dist(post[item], pgrid)
        out[item]['SD'] = compute_sd_of_dist(post[item], pgrid, Nitem)
    return out

