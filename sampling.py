'''Bootstrap: Resampling Methods

Issues
------

1. Right now parallelization only works in some scenarios,
   it hangs up on the statsmodels OLS test

Author:  Jake Torcasso
License: BSD (3-clause)

'''
from __future__ import division

import numpy as np
import pandas as pd
from inference import shift, normal
from multiprocessing import Pool
import functools
import warnings

def draw_index(data, size, by=None, seed=1234):
    '''sample indices from data

    Parameteters
    ------------
    data : array or dataframe
        data to be resampled
    size : int
        number of samples
    by : list
        list of columns in data to sample within, i.e.
        stratified random sampling within blocks defined
        by selected columns
    seed : int
        seed for pseudo-random number generator
    '''
    np.random.seed(1234)

    data = pd.DataFrame(data)

    N = len(data)

    if by is not None:

        assert isinstance(by, list)

        if not set(by).issubset(data.columns):
            raise ValueError('{} not in data columns'.format(by))

        cells = data[by].drop_duplicates().as_matrix()
        
        indices = []
        for i in range(size):
            reindex = np.array([])
            for c in cells:
                cell_index = (data[by]==c).all(axis=1)
                N_ = cell_index.sum()
                reindex = np.hstack((reindex, np.random.choice(\
                                     data[cell_index].index,size=N_)))
            indices.append(reindex)

        return indices

    else:
        return [np.random.choice(data.index, \
                                 size=N) for i in range(size)]


def bootstrap(f, data, size, fargs={}, by=None, **kwargs):
    '''construct a boostrap distribution of parameters

    Parameters
    ----------
    f : function
        function which produces a parameter from data.
    data : dict
        data to resample
    size : int
        number of bootstrap draws
    fargs : dict
        other keyword arguments for f
    by : list
        list of columns in data to statify sampling on

    kwargs
    ------
    seed : int
        seed for pseudo-random number generator
    threads : int
        number of threads to spread bootstrap on,
        default 1

    Returns
    -------
    sample : BootstrapResult instance
        object for exploring bootstrap distribution, including
        hypothesis testing

    Ex: 

    >>> def f(data, missing='drop'):
        
            endog = data['income']
            exog = data['ability']

            fit = sm.OLS(endog,exog,missing=missing).fit()

            return fit.params

    >>> bstrap = bootstrap(f, data, 1000, fargs={'missing':'drop'})

    '''

    seed = 1234 if 'seed' not in kwargs else kwargs['seed']
    threads = 1 if 'threads' not in kwargs else kwargs['threads']

    indices = draw_index(data, size, by=by, seed=seed)

    point = f(data, **fargs)

    if isinstance(point, (pd.Series, pd.DataFrame, np.ndarray)):
        pass
    elif isinstance(point, (int,float,np.int,np.float)):
        point = np.array(point)
    else:
        raise ValueError('function returns unsupported type {}'\
                         .format(type(point)))

    if point.ndim == 0:
        stack = np.hstack
    elif point.ndim == 1:
        stack = np.vstack
    elif point.ndim == 2:
        stack = np.dstack
    else:
        raise ValueError('function output too high-dimensional')
    
    if isinstance(data, (pd.DataFrame, pd.Series)):
        func = function_wrapper1
    elif isinstance(data, np.ndarray):
        func = function_wrapper2
    
    outf = [point]
    if threads == 1:
        for index in indices:
            outf.append(func(f,data,fargs,index))

    else:
        warnings.warn('Parallel processing not yet stable.')
        pool = Pool(threads)
        outf.extend(pool.map(functools.partial(func,f,data,fargs),indices))
        
    outf = [out for out in outf if out is not None]
    print 'Bootstrap succeeded on {} of {} draws'.format(len(outf)-1,size)
    
    if point.ndim == 2:
        
        return BootstrapResult(stack(outf).T)

    return BootstrapResult(stack(outf))

def function_wrapper1(f, data, fargs, index):
    '''wrapper for functions in bootstrapper'''
    
    try:
        return f(data.reindex(index), **fargs)
    except:
        return None
def function_wrapper2(f, data, fargs, index):
    '''wrapper for functions in bootstrapper'''
    
    try:
        return f(data[index], **fargs)
    except:
        return None
        
class BootstrapResult(object):
    '''class for bootstrap sample inference

    Parameters
    ----------
    empf : array-like
        empirical bootstrap parameter distribution
    '''

    def __init__(self, empf):

        self.empf = empf
        self.point = empf[0]

    def get_axis(self):
        '''find proper axis for distribution
        '''

        ndim = self.empf.ndim

        if ndim == 1:
            return None
        else:
            return 0

    def std(self):
        '''wrapper for numpy.std
        '''

        return np.std(self.empf, axis=self.get_axis())

    def mean(self):
        '''wrapper for numpy.mean
        '''

        return np.mean(self.empf, axis=self.get_axis())

    def median(self):
        '''wrapper for numpy.median
        '''

        return np.median(self.empf, axis=self.get_axis())

    def pvalue(self, twosided=True, tail='right', method='shift', null=0):
        '''generate pvalue of parameter estimate

        Parameters
        ----------
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
        null : numeric type or array-like
            value of parameter under null hypothesis, either
            a scalar or array of shape like point estimate

        Returns
        -------
        pvalue : int, float, np.ndarray
            pvalue based on bootstrap inference

        '''

        axis = self.get_axis()

        if isinstance(null, np.ndarray):
            try:
                null - self.point
            except:
                message = 'null of shape {} not '.format(null.shape)
                message += 'broadcastable with point estimate'
                raise ValueError(message)
        
        if isinstance(tail, str):
            if tail not in ['left','right']:
                raise ValueError('tail must be "left" or "right"')
        elif isinstance(tail, np.ndarray):
            if tail.shape != self.point.shape:
                raise ValueError('tail not same shape as point estimate')
            if not set(tail.flat).issubset(set(['left','right'])):
                raise ValueError('all entries in tail must be "left" or "right"')
        else:
            raise ValueError('tail must be str or array of strings')
        
        if method == 'shift':

            return shift(self.empf, self.point, null, twosided, tail, axis)

        elif method == 'normal':

            return normal(self.empf, self.point, null, twosided, tail, axis)