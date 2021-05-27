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

#class TtoEleak_Planck15(residual):
#   r"""
#   T2E leakage template a la Planck
#   te'(nu1,nu2) = te(nu1,nu2)+enu(l,nu2)*tt(nu1,nu2)
#   ee'(nu1,nu2) = ee(nu1,nu2)+enu(l,nu1)*te(nu1,nu2)+enu(l,nu2)*te(nu2,nu1)
#                 +enu(l,nu1)*enu(l,nu2)*tt(nu1,nu2)
#   enu(l,nu1)=enu0(nu1)+enu2(nu1)*l**2+enu4(nu1)*l**4
#   """

#   def eval(self,enu={'100':[0.,0.,0.]},nu=None):
#       self.enul=dict()
#       for k1 in enu.keys():
#           enu_arr=np.array(enu[k1])
#           self.enul[k1]=enu_arr[0]+enu_arr[1]*self.ell**2+enu_arr[2]*self.ell**4
#       self.freq=nu
#       dcl=dict()
#       
#       for c1,f1 in enumerate(self.freq):
#           for f2 in self.freq[c1:]:
#               dcl["tt",f1,f2] = np.zeros_like(self.cl["tt",f1,f2])
#               dcl["tt",f2,f1] = np.zeros_like(self.cl["tt",f1,f2])
#               dcl["te",f1,f2] = self.enul[f2]*self.cl["tt",f1,f2]
#               dcl["ee",f1,f2] = self.enul[f2]*self.cl["te",f2,f1] + self.enul[f1]*self.cl["te",f1,f2] +self.enul[f1]*self.enul[f2]*self.cl["tt",f1,f2]
#               #dcl["ee",f1,f2] = self.elnu2*self.cl["te",f1,f2] + self.elnu1*self.cl["te",f1,f2] +self.elnu1*self.elnu2*self.cl["tt",f1,f2]
#               dcl["te",f2,f1] = dcl["te",f1,f2]#self.elnu1*self.cl["tt",f2,f1]
#               dcl["ee",f2,f1] = dcl["ee",f1,f2]#self.elnu1*self.cl["te",f1,f2] + self.elnu2*self.cl["te",f2,f1] +self.elnu1*self.elnu2*self.cl["tt",f2,f1]

#       return dcl

#class Calibration_Planck15(residual):
#   r"""
#   Calibration matrix template a la Planck
#   G^XY_nu1nu2 = 1/yp**2 {1/(2sqrt(c^XX_nu1 c^YY_nu2))+1/(2sqrt(c^XX_nu2 c^YY_nu1))}
#   cal1,cal2 are dictionaries of calibration factors:
#   cal1[XX][0,1,2]=1/yp {1/(sqrt(c^XX_nu))}
#   cal1[XX]*cal2[YY]=G^XY
#   """

#   def eval(self,cal1={'tt':[1.,1.,1.]},cal2={'tt':[1.,1.,1.]},nu=None):
#       cal=dict()
#       for k1 in cal1.keys():
#           c1=np.array(cal1[k1])[...,np.newaxis]
#           for k2 in cal2.keys():
#               c2=np.array(cal2[k2])
#               if(k1==k2):
#                   cal[k1]=c1*c2
#               else:
#                   cal['te']=c1*c2

#       self.freq=nu
#       dcl=dict()

#       
#       for i1,f1 in enumerate(self.freq):
#           for i2,f2 in enumerate(self.freq):
#               for spec in cal.keys():
#                   dcl[spec,f1,f2] = cal[spec][i1,i2]*self.cl[spec,f1,f2]

#       return dcl

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












