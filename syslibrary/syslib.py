import os
import pkg_resources
import numpy as np
from .model import Model
from itertools import product

def _get_power_file(model):
    """ File path for the named model
    """
    data_path = pkg_resources.resource_filename('syslibrary', 'data/')
    filename = os.path.join(data_path, 'cl_%s.dat'%model)

    if os.path.exists(filename):
        return filename
    raise ValueError('No template for model '+model)

def _get_power_file_yaml(model):
    """ yaml file path for the named model
    """
    data_path = pkg_resources.resource_filename('syslibrary', 'data/')
    filename = os.path.join(data_path, '%s.yaml'%model)

    if os.path.exists(filename):
        return filename
    raise ValueError('No template for model '+model)

class Multiplication_matrix(Model):
    r"""
    first attemp to build a class for calibration
    """
    def eval(self,cXnu1=1.,cYnu2=1.):
        """
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

class residual(Model):
    r"""
    residual leakage template
    Inputs:
    - ell: array of multipoles
    - spectra: dictionary of cls. Must be in the following format:
               spectra[spec,f1,f2], with e.g., spec=tt,te,ee,etc and f1,f2=freqs
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
        name=kwargs['rootname']
        self.filename=_get_power_file_yaml(name)

        import yaml

        with open(self.filename) as file:
            self.doc = yaml.full_load(file)

    def eval(self,amp=1.,ell=None):

        dcl=dict()

        for spec in self.doc.keys():
            for i1,f1 in enumerate(self.doc[spec].keys()):
                for i2,f2 in enumerate(self.doc[spec][f1].keys()):
                    dcl[spec,f1,f2] = amp*np.array(self.doc[spec][f1][f2])[ell]

        return dcl












