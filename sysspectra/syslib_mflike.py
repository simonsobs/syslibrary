from .syslib import *
import numpy as np

class TtoEleak_Planck15(residual):
    r"""
    T2E leakage template a la Planck
    te'(nu1,nu2) = te(nu1,nu2)+enu(l,nu2)*tt(nu1,nu2)
    ee'(nu1,nu2) = ee(nu1,nu2)+enu(l,nu1)*te(nu1,nu2)+enu(l,nu2)*te(nu2,nu1)
                  +enu(l,nu1)*enu(l,nu2)*tt(nu1,nu2)
    enu(l,nu1)=enu0(nu1)+enu2(nu1)*l**2+enu4(nu1)*l**4
    """

    def eval(self,enu={'100':[0.,0.,0.]},nu=None):
        self.enul=dict()
        for k1 in enu.keys():
            enu_arr=np.array(enu[k1])
            self.enul[k1]=enu_arr[0]+enu_arr[1]*self.ell**2+enu_arr[2]*self.ell**4
        self.freq=nu
        dcl=dict()
        
        for c1,f1 in enumerate(self.freq):
            for f2 in self.freq[c1:]:
                dcl["tt",f1,f2] = np.zeros_like(self.cl["tt",f1,f2])
                dcl["tt",f2,f1] = np.zeros_like(self.cl["tt",f1,f2])
                dcl["te",f1,f2] = self.enul[f2]*self.cl["tt",f1,f2]
                dcl["ee",f1,f2] = self.enul[f2]*self.cl["te",f2,f1] + self.enul[f1]*self.cl["te",f1,f2] +self.enul[f1]*self.enul[f2]*self.cl["tt",f1,f2]
                #dcl["ee",f1,f2] = self.elnu2*self.cl["te",f1,f2] + self.elnu1*self.cl["te",f1,f2] +self.elnu1*self.elnu2*self.cl["tt",f1,f2]
                dcl["te",f2,f1] = dcl["te",f1,f2]#self.elnu1*self.cl["tt",f2,f1]
                dcl["ee",f2,f1] = dcl["ee",f1,f2]#self.elnu1*self.cl["te",f1,f2] + self.elnu2*self.cl["te",f2,f1] +self.elnu1*self.elnu2*self.cl["tt",f2,f1]

        return dcl

class Calibration_Planck15(residual):
    r"""
    Calibration matrix template a la Planck
    G^XY_nu1nu2 = 1/yp**2 {1/(2sqrt(c^XX_nu1 c^YY_nu2))+1/(2sqrt(c^XX_nu2 c^YY_nu1))}
    cal1,cal2 are dictionaries of calibration factors:
    cal1[XX][0,1,2]=1/yp {1/(sqrt(c^XX_nu))}
    cal1[XX]*cal2[YY]=G^XY
    """

    def eval(self,cal1={'tt':[1.,1.,1.]},cal2={'tt':[1.,1.,1.]},nu=None):
        cal=dict()
        for k1 in cal1.keys():
            c1=np.array(cal1[k1])[...,np.newaxis]
            for k2 in cal2.keys():
                c2=np.array(cal2[k2])
                if(k1==k2):
                    cal[k1]=c1*c2
                else:
                    cal['te']=c1*c2

        self.freq=nu
        dcl=dict()

        
        for i1,f1 in enumerate(self.freq):
            for i2,f2 in enumerate(self.freq):
                for spec in cal.keys():
                    dcl[spec,f1,f2] = cal[spec][i1,i2]*self.cl[spec,f1,f2]

        return dcl
