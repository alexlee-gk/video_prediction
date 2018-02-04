import time

import numpy as np
from tensorflow.python.util import nest


def cem(f, theta_mean, theta_cov, batch_size, n_iters,
        elite_frac=0.05, n_elite_min=10, ret_best=True, verbose=True):
    """
    Cross-entropy method.

    Args:
        f: A function mapping from theta vector to either batched scalars or
            a tuple of batched scalars and other batched tensors.
        theta_mean: Initial mean.
        theta_cov: Initial covariance.
        batch_size: Number of samples of theta to evaluate per iteration.
        iters: Number of iterations.
        elite_frac: Fraction of the best samples to keep at each iteration.
        n_elite_min: Minimum number of best samples to keep at each iteration.
        ret_best: If true, return the theta vector that achieved the best
            score in any iteration and the corresponding returned value from
            the function f. If false, return the mean of the last elite set
            (as in the original algorithm).
    """
    n_elite = max(int(np.round(batch_size * elite_frac)), n_elite_min)
    theta_best = None
    score_best = None
    arg_score_best = None
    for iter in range(n_iters):
        start_time = time.time()
        if verbose:
            print('cem iteration %d' % iter)
        thetas = np.random.multivariate_normal(theta_mean, theta_cov, batch_size)
        scores = f(thetas)
        if isinstance(scores, tuple):
            scores, arg_scores = scores
        else:
            arg_scores = None
        elite_inds = np.argsort(scores)[:n_elite]
        elite_thetas = thetas[elite_inds]
        theta_mean = np.mean(elite_thetas, axis=0)
        theta_cov = np.cov(elite_thetas, rowvar=False)
        if theta_best is None or scores[elite_inds[0]] < score_best:
            theta_best = thetas[elite_inds[0]]
            score_best = scores[elite_inds[0]]
            if arg_scores is not None:
                arg_score_best = nest.map_structure(lambda x: x[elite_inds[0]], arg_scores)
        if verbose:
            print('time taken for iteration %d: %.2f' % (iter, time.time() - start_time))
            print('best score for iteration %d: %f' % (iter, scores[elite_inds[0]]))
            print('overall best score so far: %f' % score_best)
    if ret_best:
        if arg_scores is not None:
            score_best = (score_best, arg_score_best)
        return theta_best, score_best
    else:
        return theta_mean
