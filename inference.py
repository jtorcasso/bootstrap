'''Bootstrap: Methods of Inference

Issues
------


Author:  Jake Torcasso
License: BSD (3-clause)

'''
from __future__ import division
import numpy as np
from scipy.stats import norm

def single_pvalue(dist, point, null, twosided, tail, method):
    '''generate single pvalue of parameter estimate

    Parameters
    ----------
    dist : 1-d array
        bootstrap distribution of parameter estimates
    point : array-like
        point estimate of parameter estimate
    twosided : boolean
        False, for one-sided p-values and True for two-sided
    null : array-like
        value of parameter under null hypothesis
        array of shape like point estimate
    tail : str or array
        array of strings, 'right' 
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

def stepdown_pvalue(dist, point, null, twosided, tail, method):
    '''calculate stepdown pvalue for point estimate

    Parameters
    ----------
    dist : array-like
        bootstrap distribution of parameter estimates
    point : array-like
        point estimate of parameter estimate
    null : array-like
        parameter value under null hypothesis
    twosided : boolean
        False, for one-sided p-values and True for two-sided
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

    single_pvals = single_pvalue(dist, point, null, twosided, tail, method)
    stepdown_pvals = np.zeros(single_pvals.shape)
    
    test = (dist - dist.mean(axis=0) - null)/dist.std(axis=0)
    if not twosided:
        test[:,tail == 'left'] = -test[:,tail == 'left']

    tail = np.array(['right']).reshape((1,))

    point = (point + null)/dist.std(axis=0)
    for j in xrange(stepdown_pvals.size):
        jindex = np.argmin(single_pvals)
        jpoint = np.resize(point[jindex], 1)
        

        if twosided:
            jmaxes = np.argmax(abs(test), axis=1)
        else:
            jmaxes = np.argmax(test, axis=1)

        jtest = test[range(test.shape[0]), jmaxes]
        jpval = single_pvalue(jtest, jpoint, 0, twosided, tail, method)
        jpval = shift2(jtest, jpoint, 0, twosided, tail)
        stepdown_pvals[jindex] = jpval

        single_pvals[jindex] = 1.1
        test = test[:,[i for i in xrange(test.shape[1]) if i != jindex]]

    return stepdown_pvals


def shift2(dist, point, null, twosided=True, tail='right'):
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
        specify 'left' or 'right', an array of 
        such strings for one-sided tests. 'right' 
        implies a one-sided test that the point
        estimate is greater than the null

    Returns
    -------
    pvalue : array-like
        pvalues, of shape similar to point estimate
    '''

    # setting null distribution as dist centered at null
    nulldist = dist
    N = len(nulldist)
    
    if twosided:
        right_tail = (nulldist >= abs(point - null) + null).sum(axis=0)
        left_tail = (nulldist < -abs(point - null) + null).sum(axis=0)

        return (right_tail + left_tail)/N
    
    else:
        
        right_tail = np.resize((nulldist >= point).sum(axis=0), point.size)
        left_tail = np.resize((nulldist <= point).sum(axis=0), point.size)
        
        pvalue = np.zeros(point.shape)
        pvalue[tail == 'left'] = left_tail[tail == 'left']
        pvalue[tail == 'right'] = right_tail[tail == 'right']
        return pvalue/N

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
        specify 'left' or 'right', an array of 
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
        
        right_tail = np.resize((nulldist >= point).sum(axis=0), point.size)
        left_tail = np.resize((nulldist <= point).sum(axis=0), point.size)
        
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
        specify 'left' or 'right', an array of 
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

        pvalue = np.zeros(point.shape)
        pvalue[tail == 'left'] = left_tail[tail == 'left']
        pvalue[tail == 'right'] = right_tail[tail == 'right']
        return pvalue

class BootstrapResult(object):
    '''class for bootstrap sample inference

    Parameters
    ----------
    empf : array-like
        empirical bootstrap parameter distribution
    '''

    def __init__(self, empf, shape):

        self.empf = empf[1:]
        self.point = empf[0]
        self.shape = shape

    def std(self):
        '''wrapper for numpy.std
        '''

        return self._format(np.std(self.empf, axis=0))

    def mean(self):
        '''wrapper for numpy.mean
        '''

        return self._format(np.mean(self.empf, axis=0))

    def median(self):
        '''wrapper for numpy.median
        '''

        return self._format(np.median(self.empf, axis=0))

    def _format(self, stat):
        '''reshapes output to conform to function output

        Parameters
        ----------
        stat : array-like
            statistic to reshape

        Returns
        -------
        stat_formatted : array-like
            reshaped statistic
        '''
        return stat.reshape(self.shape)

    def pvalue(self, twosided=True, null=0, **kwargs):
        '''generate pvalue of parameter estimate

        Parameters
        ----------
        twosided : boolean
            False, for one-sided p-values and True for two-sided
        null : numeric type or array-like
            value of parameter under null hypothesis, either
            a scalar or array of shape like point estimate

        kwargs
        ------
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

        size = self.point.size

        tail = np.resize(np.array(kwargs.get('tail', 'right')), size)
        method = kwargs.get('method', 'shift')
        mtp = kwargs.get('mtp', None)

        null = np.resize(np.array(null), size)

        if not isinstance(null[0], (int,float,np.float,np.int)):
            raise ValueError('null should be int or float, received {}'\
                             .format(type(null[0])))

        if not set(tail.flat).issubset(set(['left','right'])):
            raise ValueError('all entries in tail must be "left" or "right"')

        if method not in ['shift', 'normal']:
            raise ValueError('method must be either "shift" or "normal"')

        if mtp is not None:
            if mtp not in ['stepdown']:
                raise ValueError('mtp must be None or "stepdown"')

        if (mtp is None) or (self.shape == ()):
            return self._format(single_pvalue(self.empf, self.point, \
                                 null, twosided, tail, method))
        elif mtp == 'stepdown':
            return self._format(stepdown_pvalue(self.empf, self.point, \
                                 null, twosided, tail, method))