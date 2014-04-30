'''Bootstrap: Methods of Inference

Issues
------


Author:  Jake Torcasso
License: BSD (3-clause)

'''
from __future__ import division
import numpy as np
from scipy.stats import norm

def single_pvalue(dist, point, twosided, tail, method, null):
    '''generate single pvalue of parameter estimate

    Parameters
    ----------
    dist : 1-d array
        bootstrap distribution of parameter estimates
    point : array-like
        point estimate of parameter estimate
    twosided : boolean
        False, for one-sided p-values and True for two-sided
    null : numeric type or array-like
        value of parameter under null hypothesis, either
        a scalar or array of shape like point estimate
    tail : str or array
        specify 'left' or 'right', or an array of 
        such strings for one-sided tests. 'right' 
        implies a one-sided test that the point
        estimate is greater than the null
    method : str
        'normal', assume null distribution is normal
        'shift', assume only that null distribution is the 
        bootstrap distribution recentered at the null
    mtp : None or str
        None, for no multiple testing procedure, 'stepdown', 
        for the stepdown procedure

    Returns
    -------
    pvalue : int, float, np.ndarray
        pvalue based on bootstrap inference

    Notes
    -----
    Calls other functions
    '''

    
    if method == 'shift':

        return shift(dist, point, null, twosided, tail)

    elif method == 'normal':

        return normal(dist, point, null, twosided, tail)     

def stepdown_pvalue(dist, point, twosided, tail, method, null):
    '''calculate stepdown pvalue for point estimate

    Parameters
    ----------
    dist : array-like
        bootstrap distribution of parameter estimates
    point : array-like
        point estimate of parameter estimate
    twosided : boolean
        False, for one-sided p-values and True for two-sided
    null : numeric type or array-like
        value of parameter under null hypothesis, either
        a scalar or array of shape like point estimate
    tail : str or array
        specify 'left' or 'right', or an array of 
        such strings for one-sided tests. 'right' 
        implies a one-sided test that the point
        estimate is greater than the null
    method : str
        'normal', assume null distribution is normal
        'shift', assume only that null distribution is the 
        bootstrap distribution recentered at the null
    mtp : None or str
        None, for no multiple testing procedure, 'stepdown', 
        for the stepdown procedure

    Returns
    -------
    pvalue : int, float, np.ndarray
        pvalue based on bootstrap inference

    '''

    single_pvals = single_pvalue(dist, point, twosided, tail, method, null)
    stepdown_pvals = np.zeros(single_pvals.shape)

    for j in xrange(stepdown_pvals.size):
        jmin = np.argmin(single_pvals)

    return None


def shift(dist, point, null, twosided=True, tail='right'):
    '''hypothesis testing using shift method

    Parameters
    ----------
    dist : array-like
        parameter distribution
    point : array-like
        point estimate of parameter
    null : array-like
        parameter value under null hypothesis
    twosided : boolean
        if True, calculates two-sided p-values
    tail : str or array
        specify 'left' or 'right', or an array of 
        such strings for one-sided tests. 'right' 
        implies a one-sided test that the point
        estimate is greater than the null

    Returns
    -------
    pvalue : array-like
        pvalues, of shape similar to point estimate
    '''

    # setting null distribution as dist centered at null
    nulldist = dist - dist.mean(axis=0, keepdims=True) + null
    N = len(nulldist)
    
    if twosided:
        right_tail = (nulldist >= abs(point - null) + null).sum(axis=0)
        left_tail = (nulldist < -abs(point - null) + null).sum(axis=0)

        return (right_tail + left_tail)/N
    
    else:
        
        right_tail = (nulldist >= point).sum(axis=0)
        left_tail = (nulldist <= point).sum(axis=0)
        
        if isinstance(tail, str):
            if tail == 'left':
                return left_tail/len(nulldist)
            elif tail == 'right':
                return right_tail/N
        elif isinstance(tail, np.ndarray):
            pvalue = np.zeros(point.shape)
            pvalue[tail == 'left'] = left_tail[tail == 'left']
            pvalue[tail == 'right'] = right_tail[tail == 'right']
            return pvalue/N
        
        
def normal(dist, point, null, twosided=True, tail='right'):
    '''hypothesis testing assuming normal distribution

    Parameters
    ----------
    dist : array-like
        parameter distribution
    point : array-like
        point estimate of parameter
    null : array-like
        parameter value under null hypothesis
    twosided : boolean
        if True, calculates two-sided p-values
    tail : str or array
        specify 'left' or 'right', or an array of 
        such strings for one-sided tests. 'right' 
        implies a one-sided test that the point
        estimate is greater than the null

    Returns
    -------
    pvalue : array-like
        pvalues, of shape similar to point estimate
    '''
    
    if twosided:
        return 2*norm.sf(abs(point - null)/dist.std(axis=0))
    else:
        left_tail = norm.cdf((point - null)/dist.std(axis=0))
        right_tail = norm.sf((point - null)/dist.std(axis=0))
        if isinstance(tail, str):
            if tail == 'left':
                return left_tail
            else:
                return right_tail
        elif isinstance(tail, np.ndarray):
            pvalue = np.zeros(point.shape)
            pvalue[tail == 'left'] = left_tail[tail == 'left']
            pvalue[tail == 'right'] = right_tail[tail == 'right']
            return pvalue
