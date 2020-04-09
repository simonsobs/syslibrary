import numpy as np
from .model import Model

class Multiplication_matrix(Model):
	r"""
	first attemp to build a class for calibration
	"""
	def eval(self,cXnu1=1.,cYnu2=1.):
		"""
		comments here
		Note that the order of cXnu1,cYnu2 matters.
		This follows Eq.(37) in Planck 2018 V.Likelihood paper
		"""
		cXnu1=np.array(cXnu1)[...,np.newaxis]
		cYnu2=np.array(cYnu2)
		return cXnu1*cYnu2

class Calibration(Multiplication_matrix):
	r""" 
	Alias of :class:`Multiplication_matrix`
	"""
	pass