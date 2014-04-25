'''Bootstrap: Methods of Inference

Issues
------


Author:  Jake Torcasso
License: BSD (3-clause)

'''

import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import norm

def run_in_parallel(f, *args):
	'''
	run a function on sets of data simultaneously
	

	Parameters
	----------
	f : function
		function to run on data
	args : tuple
		each element a data array to feed into f

	Returns
	-------
	results : list
		list of results from f on args
	'''

	pool = Pool()
	results = pool.map(f, *args)
	pool.close()

	return results

class bootstrap(object):
	'''
	class for bootstrapping parameters


	Parameters
	----------
	X : array
		data matrix
	f : function
		function producing statistic(s) for X; f(X) --> B
	args : tuple
		arguments to pass onto f
	draws : int
		number of bootstrap draws
	'''

	def __init__(self, X, f, args=(), draws = 1000):

		check_data = [isinstance(X, i) for i in [np.ndarray, pd.DataFrame, pd.Series]]

		assert(True in check_data)
		assert(hasattr(f, '__call__'))

		self.X = X
		self.f = f
		self.args = args
		self.draws = draws
		self.indices = self.draw_sample_indices()
		self.check_function()
		self.get_distribution()

	def check_function(self, B = None):
		'''
		recursively checks the output of self.f to ensure:
		1) if np.ndarray, ndim == 1 and elements are numeric
		2) else, ensures output is numeric

		Parameters
		----------
		B : object
			object to check type

		'''

		if B is None:

			B = self.f(self.X, *self.args)

			self.f_outputs_array = isinstance(B, np.ndarray)
			self.f_output_ndim = 0 if not self.f_outputs_array else B.ndim
			self.f_output_shape = None if not self.f_outputs_array else B.shape
			self.f_output_type = type(B)

		accepted_types = [int, float, long, np.ndarray]

		checks = [isinstance(B, _t) for _t in accepted_types]

		assert(True in checks)

		if isinstance(B, np.ndarray):

			assert(B.ndim == 1)

			for	b in B:

				self.check_function(b)

	def draw_sample_indices(self):
		'''
		draws random index arrays for bootstrap sampling


		Returns
		-------
		indices : list
			list of arrays which index the bootstrap samples
		'''
		draws = self.draws

		X = self.X

		N = len(X)

		if isinstance(X, np.ndarray):
			return [np.random.choice(range(N), N) for i in range(draws)]
		else:
			return [np.random.choice(X.index, N) for i in range(draws)]


	def draw_sample(self, indices):
		'''
		draws a bootstrap sample

		Parameters
		----------
		indices : list
			list of arrays, each array indexes a bootstrap sample


		Returns
		-------
		sample : array
			data matrix
		'''

		if isinstance(self.X, np.ndarray):
			return self.X[indices,:]
		else:
			X = self.X.reindex(indices)
			X.index = range(len(X))
			return X

	def get_distribution(self):
		'''
		generates boostrap distribution of estimated parameters


		Returns
		-------
		B : int, float, np.ndarray
			parameter distribution(s), estimated from self.f(self.X)
		'''
		
		X = self.X

		f = self.f

		self.point_estimate = f(X, *self.args)

		ndim = self.f_output_ndim

		errors = 0
		if ndim == 0:
			
			for i,index in enumerate(self.indices):
				X = self.draw_sample(index)
				try:
					Bnew = f(X, *self.args)
				except:
					errors += 1
					continue
				try:
					B = np.hstack((B, Bnew))
				except:
					B = np.array(Bnew)
		
		elif ndim == 1:

			for i,index in enumerate(self.indices):
				X = self.draw_sample(index)
				try:
					Bnew = f(X, *self.args)
				except:
					errors += 1
					continue
				try:
					B = np.vstack((B, Bnew))
				except:
					B = Bnew
			B = B.transpose()

		if errors > 0:
			print "Bootstrap estimation succeeded on %d of %d draws." % (self.draws - errors, self.draws)
		self.Bdist = B
		return B

	def get_point_estimate(self):
		'''point estimates of parameters
		'''
		return self.point_estimate

	def apply_to_distribution(self, stat_func):
		'''
		applies a statistical function to the bootstrap distribution
		
		stat_func : function
			function to transform empirical distribution of data,
			must operate on an array and return point estimate

		Returns
		-------
		B : int, float, np.ndarray
			number or array
		'''

		B = self.Bdist

		stat = stat_func
		ndim = self.f_output_ndim

		if ndim == 0:
			return stat_func(B)
		elif ndim == 1:
			if 'axis' in stat_func.func_code.co_varnames:
				B = stat_func(B, axis = ndim)
			else:
				print "Warning: can't apply axis to stat_func"
				B = stat_func(B)

			B = B.reshape(self.f_output_shape)

			return B

	def get_std(self):
		'''
		generates the bootstrapped standard error

		Returns
		-------
		std : int, float, np.ndarray
			the standard error of parameter point estimate(s)
		'''

		return self.apply_to_distribution(np.std)

	def get_pvalue(self, null=0, twosided=False, method='normal'):
		'''generate pvalue of parameter estimate

		Parameters
		----------
		null : int, float, np.ndarray
			the expected value of the parameter(s) under the null. If
			an np.ndarray must match shape of output from self.f
		twosided : boolean
			False, for one-sided p-values and True for two-sided
		method : str
			method to construct p-values
			'normal', assume null distribution is normal
			'nonparametric', assume only that null distribution is the 
			bootstrap distribution recentered at the null

		Returns
		-------
		pvalues : int, float, np.ndarray
			p-value of parameter(s)


		Notes
		-----
		Pay close attention to np.ndarray broadcasting rules

		'''

		
		if method == 'nonparametric':

			def f(Bdist, axis=None):

				mean = Bdist.mean(axis=axis)
				Bpoint = self.point_estimate

				# Reshaping to comply with broadcasting properties
				if isinstance(self.point_estimate, np.ndarray):
					Bpoint = self.point_estimate.reshape((-1,1))
					mean = mean.reshape((-1,1))


				Bnull = Bdist - mean # recentering distribution
				
				if twosided:
					absBpoint = abs(Bpoint)
					pvalueNeg = (Bnull < -absBpoint).sum(axis=axis)/float(Bnull.shape[-1])

					pvaluePos = (Bnull >= absBpoint).sum(axis=axis)/float(Bnull.shape[-1])

					pvalues = pvalueNeg + pvaluePos				
				
				else:
					pvalueNeg = (Bnull < Bpoint).sum(axis=axis)/float(Bnull.shape[-1])
					pvaluePos = (Bnull >= Bpoint).sum(axis=axis)/float(Bnull.shape[-1])
					
					if isinstance(Bpoint, np.ndarray):
						pvalueNeg[Bpoint.flat >= null] = 0
						pvaluePos[Bpoint.flat < null] = 0
					else:
						pvalueNeg = 0 if Bpoint >= null else pvalueNeg
						pvaluePos = 0 if Bpoint < null else pvaluePos

					pvalues = pvalueNeg + pvaluePos

				return pvalues

		elif method == 'normal':

			def f(Bdist, axis=None):
				Bpoint = self.point_estimate
				sides = 2 if twosided else 1
				return sides*norm.sf((Bpoint - null)/Bdist.std(axis=axis))

		

		return self.apply_to_distribution(f)
	











# Test Code

if __name__ == '__main__':

	np.random.seed(1234)

	N = 1500
	a = 10 + 10*np.random.randn(N)
	b = 20 + 20*np.random.randn(N)
	c = np.random.randn(N)
	d = 32 + 16*np.random.randn(N)
	constant = np.ones(N)

	import statsmodels.api as sm

	X = np.vstack((a,constant,b,c,d))
	X = X.transpose()
	y = X[:,0]
	x = X[:,1:]

	model = sm.OLS(y,x)

	results = model.fit()

	print results.summary()

	def simpleOLS(X):
		y = X[:,0]
		x = X[:,1:]
		
		results = sm.OLS(y,x).fit()

		B = results.params

		return B

	boot = bootstrap(X, simpleOLS)

	print boot.get_point_estimate()
	print boot.get_pvalue(twosided=True)
	print boot.get_pvalue(twosided=True, method='nonparametric')
	print boot.get_pvalue(twosided=False)
	print boot.get_pvalue(twosided=False, method='nonparametric')

	def mean(X):
		X = X.as_matrix()

		return simpleOLS(X)

	boot = bootstrap(pd.DataFrame(X), mean)

	print boot.get_pvalue(twosided=True)