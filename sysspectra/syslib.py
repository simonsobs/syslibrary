import os
import pkg_resources
import numpy as np
from .model import Model
from fgspectra.power import PowerSpectrumFromFile
from itertools import product

def _get_power_file(model):
    """ File path for the named model
    """
    data_path = pkg_resources.resource_filename('sysspectra', 'data/')
    filename = os.path.join(data_path, 'cl_%s.dat'%model)
    #print(filename,'we are in _get_power_file')
    if os.path.exists(filename):
        return filename
    raise ValueError('No template for model '+model)

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

class TemplateFromFile(PowerSpectrumFromFile):
    """PowerSpectrum for Thermal Sunyaev-Zel'dovich (Dunkley et al. 2013)."""

    def __init__(self,**kwargs):
    	names=kwargs['filenames']#['genericTemplateTT']
    	filenames=[]
    	for i,n in np.ndenumerate(names):
    		print(i)
    		"""Intialize object with parameters."""
    		filenames.append(_get_power_file(n))
    	print(filenames,'we are in TemplateFromFile')

    	super().__init__(filenames)

class TemplatesFromFiles(PowerSpectrumFromFile):
    """PowerSpectrum for Thermal Sunyaev-Zel'dovich (Dunkley et al. 2013)."""

    def __init__(self,**kwargs):
    	freqs=kwargs['nu']
    	nfreqs=len(freqs)
    	corr=product(freqs,freqs)
    	rootname='generic_template_'
    	filenames=[]
    	for i,c in enumerate(corr):
    		#print(i,c)
    		#idx = (i%nfreqs, i//nfreqs)
    		name=rootname+c[0]+'_'+c[1]
    		"""Intialize object with parameters."""
    		filenames.append(_get_power_file(name))
    	filenames=np.reshape(filenames,(nfreqs,nfreqs))
    	#print(filenames,'we are in TemplateFromFile')

    	super().__init__(filenames) 