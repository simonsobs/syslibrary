import os
import pkg_resources
import numpy as np
from .model import Model
#from fgspectra.power import PowerSpectrumFromFile
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

def _get_power_file_yaml(model):
    """ File path for the named model
    """
    data_path = pkg_resources.resource_filename('sysspectra', 'data/')
    filename = os.path.join(data_path, '%s.yaml'%model)
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

# class TemplateFromFile(PowerSpectrumFromFile):
#     """PowerSpectrum for Thermal Sunyaev-Zel'dovich (Dunkley et al. 2013)."""

#     def __init__(self,**kwargs):
#         names=kwargs['filenames']#['genericTemplateTT']
#         filenames=[]
#         for i,n in np.ndenumerate(names):
#             print(i)
#             """Intialize object with parameters."""
#             filenames.append(_get_power_file(n))
#         print(filenames,'we are in TemplateFromFile')

#         super().__init__(filenames)

# class TemplatesFromFiles(PowerSpectrumFromFile):
#     """PowerSpectrum for Thermal Sunyaev-Zel'dovich (Dunkley et al. 2013)."""

#     def __init__(self,**kwargs):
#         freqs=kwargs['nu']
#         nfreqs=len(freqs)
#         corr=product(freqs,freqs)
#         rootname='generic_template_'
#         filenames=[]
#         for i,c in enumerate(corr):
#             #print(i,c)
#             #idx = (i%nfreqs, i//nfreqs)
#             name=rootname+c[0]+'_'+c[1]
#             """Intialize object with parameters."""
#             filenames.append(_get_power_file(name))
#         filenames=np.reshape(filenames,(nfreqs,nfreqs))
#         #print(filenames,'we are in TemplateFromFile')

#         super().__init__(filenames) 

class residual(Model):
    r"""
    residual leakage template
    Inputs:
    - ell: array of multipoles
    - spectra: dictionary of cls. Must be in the following format:
               spectra[spec,f1,f2], with spec=tt,te,ee and f1,f2=freqs
    """
    def __init__(self,**kwargs):
        self.ell=kwargs['ell']
        self.cl=kwargs["spectra"]
        self.set_defaults(**kwargs)

    def eval(self,amp=1.):

        for k in self.cl.keys():
            self.cl[k]*=amp

        return self.cl

class ReadTemplateFromFile(Model):
    """PowerSpectrum for generic template read from yaml file"""

    def __init__(self,**kwargs):
        name=kwargs['rootname']#['genericTemplateTT']
        self.filename=_get_power_file_yaml(name)

        #super().__init__(self.filename)

    def eval(self,amp={'field1':np.ones((3,3))},ell=None):
    	#amp={'tt':np.ones((3,3)),
    	#'te':np.ones((3,3)),'ee':np.ones((3,3))},ell=None):
        import yaml

        dcl=dict()

        with open(self.filename) as file:
            doc = yaml.full_load(file)
            for spec in doc.keys():
                for i1,f1 in enumerate(doc[spec].keys()):
                    for i2,f2 in enumerate(doc[spec][f1].keys()):
                        dcl[spec,f1,f2] = amp[spec][i1,i2]*np.array(doc[spec][f1][f2])[ell]

        return dcl












