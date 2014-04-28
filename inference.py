'''Bootstrap: Methods of Inference

Issues
------


Author:  Jake Torcasso
License: BSD (3-clause)

'''
from __future__ import division
import numpy as np
from scipy.stats import norm
        

def shift(dist, point, null, twosided=True, tail='right', axis=None):
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
    axis : None or int
        axis average over    

    Returns
    -------
    pvalue : array-like
        pvalues, of shape similar to point estimate
    '''

    # setting null distribution as dist centered at null
    nulldist = dist - dist.mean(axis=axis, keepdims=True) + null
    N = len(nulldist)
    
    if twosided:
        right_tail = (nulldist >= abs(point - null) + null).sum(axis=axis)
        left_tail = (nulldist < -abs(point - null) + null).sum(axis=axis)

        return (right_tail + left_tail)/N
    
    else:
        
        right_tail = (nulldist >= point).sum(axis=axis)
        left_tail = (nulldist <= point).sum(axis=axis)
        
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
        
        
def normal(dist, point, null, twosided=True, tail='right', axis=None):
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
    axis : None or int
        axis average over    

    Returns
    -------
    pvalue : array-like
        pvalues, of shape similar to point estimate
    '''
    
    if twosided:
        return 2*norm.sf(abs(point - null)/dist.std(axis=axis))
    else:
        left_tail = norm.cdf((point - null)/dist.std(axis=axis))
        right_tail = norm.sf((point - null)/dist.std(axis=axis))
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
