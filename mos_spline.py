import abc
import numbers

import numpy
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
import sklearn.base
import sklearn.utils.validation
import pandas as pd
from . import base
import prob_spline
import matplotlib.pyplot as pyplot
import joblib
from time import gmtime, strftime

#msq_file = "Vector_Data(NoZeros).csv"

class MosSpline():
	'''
	Utilizing the PoissonSpline code and a datafile consisting of
	the sampled mosquito counts to generate the individual splines for each bird
	
	NOTE : We do not intend to use this in our final model, rather as a method to test the ODE
	'''


	def __init__(self, data_file,sigma = 0, period=prob_spline.period(), sample = 0):

		msg = 'datafile must be a string'
		assert isinstance(data_file, str), msg
		assert (sigma >= 0), 'sigma must be nonnegative.'
		assert (n_samples >= 0), 'number of samples must be nonnegative.'
		assert isinstance(n_samples,int), 'number of samples must be an integer'
		
		self.data_file = data_file

		self.read_data()

		self.X=prob_spline.time_transform(self.time)

		if sample==1:
			self.generate_samples()
			self.splines = self.get_host_splines(self.X,self.samples,sigma,period)
		else:
			self.splines = self.get_host_splines(self.X,self.Y,sigma,period)

	def read_data(self):

		count_data = pd.read_csv(self.data_file,index_col=0)
		self.birdnames = count_data.index
		self.time = numpy.array([int(x) for x in count_data.columns])
		self.Y = count_data.as_matrix()
		return()
	
	def get_host_splines(self,X,Y_mat,sigma,period):
		Y = numpy.squeeze(Y_mat.T)
		poisson_spline = prob_spline.PoissonSpline(sigma = sigma, period=period)
		poisson_spline.fit(X, Y)
		return(poisson_spline)

	def evaluate(self,X,index=0):			# Evaluate the splines at given values X
		return(numpy.array(self.splines(X)))

	__call__ = evaluate

	def log_derivative(self,X):
		return(numpy.array(self.splines.log_derivative(X)))

	def pos_der(self,X):
		return(numpy.array(numpy.max((self.derivative(X),0))))
	
	def neg_der(self,X):
		return(numpy.array(numpy.min((self.derivative(X),0))))
	

	def plot(self):
		'''
		A function to plot the data and spline fit of the specified species
		Defaults to all species given, but allows for input of specified species index
		'''
		x = numpy.linspace(numpy.min(self.X), numpy.max(self.X), 1001)
		handles = []
		s = pyplot.scatter(prob_spline.inv_time_transform(self.X), self.Y, color = 'black',
	                   label = 'Cs. melanura')
		handles.append(s)
		l = pyplot.plot(prob_spline.inv_time_transform(x), self.splines(x),
			label = 'Fitted PoissonSpline($\sigma =$ {:g})'.format(self.splines.sigma))
		handles.append(l[0])
		pyplot.xlabel('$x$')
		pyplot.legend(handles, [h.get_label() for h in handles],fontsize = 'xx-small',loc=0)
		pyplot.show()
		return()

	def generate_samples(self):
		self.samples = numpy.random.poisson(lam=self.Y,size = (len(self.Y.T))) 
		return()

