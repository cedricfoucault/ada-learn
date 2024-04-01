"""
Implementation of the reduced Bayesian model learner algorithm as described
in Nassar et al. (2012) online methods.
"""
import numpy as np
import scipy.stats

P_C = 1 / 10.
STD_GEN = 10 / 300.
B_0 = 0.5
TAU_0 = 0.5

def run_inference(outcome,
    p_c=P_C,
    std_gen=STD_GEN,
    b_0=B_0,
    tau_0=TAU_0):
    """
    Runs the inference algorithm on the given 1-dimensional sequence of outcomes.

    Parameters
    ----------
    outcome: 1D np.array of shape (n_trials,)
    std_gen: the standard deviation of the generative distribution of outcomes
    b_0: the prior value of the learner's mean estimate
        (i.e., the mean of the posterior distribution on the generative mean)
    tau_0: the prior value of the learner's relative uncertainty (see below)
    
    Returns
    -------
    inf_dict: a dictionary containing the latent variables of the model
    inference at each time step, given the oberved outcomes including the one
    at the current time step.

    The dict contains the following variables for the following keys:
    - 'mean' (called B in the paper):
       Mean of the posterior distribution in the mean of the generative distribution
    - 'relative_uncertainty' (called tau in the paper):
       Relative uncertainty
       = (uncertainty about the location of the generative mean
         / uncertainty about the location of the next outcome)
       = (variance of the posterior distribution in the generative mean
         / variance of the posterior distribution in the next outcome)
    - 'cpp' (called omega in the paper):
        (inferred) probability of a change point having just occurred.
    """
    assert outcome.ndim == 1, "Expect a 1-dimensional sequence of outcomes"
    unif_mean_density = 1 # here we assume the mean is uniformly sampled in [0, 1]
    var_gen = std_gen ** 2

    n_trials = outcome.shape[0]
    b = np.empty(n_trials + 1) # include the prior the inference loop
    tau = np.empty(n_trials + 1)
    b[0] = b_0
    tau[0] = tau_0
    omega = np.empty(n_trials)
    outcome_probability = np.empty(n_trials)
    for t in range(n_trials):
        # Variance of the posterior distribution in the next outcome
        # Nassar et al. (2012) eq (6)
        sigma_sq = var_gen + tau[t] / (1 - tau[t]) * var_gen
        # outcome probability under prev. belief: p(X_t | no_cp)
        outcome_probability[t] = scipy.stats.norm.pdf(outcome[t],
            loc=b[t], scale=np.sqrt(sigma_sq))
        # Probability of a change point
        # Nassar et al. (2012) eq (5)
        omega_num = p_c * unif_mean_density
        omega_denom = omega_num + outcome_probability[t] * (1 - p_c)
        omega[t] = (omega_num / omega_denom)
        # Update in relative uncertainty
        # Nassar et al. (2012) eq (7) (corrected)
        tau_term_1 = var_gen * omega[t]
        tau_term_2 = (1 - omega[t]) * tau[t] * var_gen
        tau_term_3 = (omega[t] * (1 - omega[t])
            * ((outcome[t] * tau[t] + b[t] * (1 - tau[t]) - outcome[t]) ** 2))
            # /!\ Note: If I add a (* var_gen) factor to term_3,
            # I seem to obtain better correlations with human learning rates
            # and such correlations better correlate with other measures of adaptability
            # across subjects. I wonder why and whether there is something to investigate
            # either on the model algorithm side or on the human behaviour side.
        tau[t+1] = ((tau_term_1 + tau_term_2 + tau_term_3)
            / (var_gen + tau_term_1 + tau_term_2 + tau_term_3))
        # Learning rate for the update in mean belief
        # Nassar et al. (2012) eq (4)
        alpha = tau[t] + (1 - tau[t]) * omega[t] # learning rate
        # Update in mean belief
        # Nassar et al. (2012) eq (3)
        delta = outcome[t] - b[t] # prediction error
        b[t+1] = b[t] + alpha * delta

    return {
        'mean': b[1:], # remove the prior from the returned value.
                       # b[i] = belief having observed outcome[i]
        'relative_uncertainty': tau[1:], # same.
        'cpp': omega,
        'outcome_probability': outcome_probability
    }

    