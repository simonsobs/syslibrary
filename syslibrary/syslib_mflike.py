from .syslib import Residual
import numpy as np


class TtoEleak_Planck15(Residual):
    r"""
    T2E leakage template a la Planck
    te'(nu1,nu2) = te(nu1,nu2)+enu(l,nu2)*tt(nu1,nu2)
    ee'(nu1,nu2) = ee(nu1,nu2)+enu(l,nu1)*te(nu1,nu2)+enu(l,nu2)*te(nu2,nu1)
                  +enu(l,nu1)*enu(l,nu2)*tt(nu1,nu2)
    enu(l,nu1)=enu0(nu1)+enu2(nu1)*l**2+enu4(nu1)*l**4
    """

    def eval(self, enu={'100': [0., 0., 0.]}, nu=None):
        self.enul = dict()
        for k1 in enu.keys():
            enu_arr = np.array(enu[k1])
            self.enul[k1] = enu_arr[0] + enu_arr[1] * self.ell ** 2 + enu_arr[2] * self.ell ** 4
        self.freq = nu
        dcl = dict()

        for c1, f1 in enumerate(self.freq):
            for f2 in self.freq[c1:]:
                dcl["tt", f1, f2] = np.zeros_like(self.cl["tt", f1, f2])
                dcl["tt", f2, f1] = np.zeros_like(self.cl["tt", f1, f2])
                dcl["te", f1, f2] = self.enul[f2] * self.cl["tt", f1, f2]
                dcl["ee", f1, f2] = self.enul[f2] * self.cl["te", f2, f1] + self.enul[f1] * self.cl["te", f1, f2] + \
                                    self.enul[f1] * self.enul[f2] * self.cl["tt", f1, f2]
                # dcl["ee",f1,f2] = self.elnu2*self.cl["te",f1,f2] + self.elnu1*self.cl["te",f1,f2] +self.elnu1*self.elnu2*self.cl["tt",f1,f2]
                dcl["te", f2, f1] = dcl["te", f1, f2]  # self.elnu1*self.cl["tt",f2,f1]
                dcl["ee", f2, f1] = dcl[
                    "ee", f1, f2]  # self.elnu1*self.cl["te",f1,f2] + self.elnu2*self.cl["te",f2,f1] +self.elnu1*self.elnu2*self.cl["tt",f2,f1]

        return dcl


class Rotation_alm(Residual):
    r"""
    apply rotation of C_ell due to polangle miscalibration
    each freq channel nu is rotated by its own polangle alpha_nu
    NB: alpha must be in deg
    """

    def eval(self, alpha=1., nu=None, cls=['tt']):
        self.freq = nu
        # NB must check that len(alpha)==len(freq)
        ca = np.cos(np.deg2rad(2. * np.array(alpha)))
        sa = np.sin(np.deg2rad(2. * np.array(alpha)))
        dcl = dict()

        for i1, f1 in enumerate(self.freq):
            for i2, f2 in enumerate(self.freq):
                dcl['te', f1, f2] = ca[i2] * self.cl['te', f1, f2]
                dcl['ee', f1, f2] = ca[i1] * ca[i2] * self.cl['ee', f1, f2]
                dcl['tt', f1, f2] = self.cl['tt', f1, f2]
                if 'bb' in cls:
                    dcl['ee', f1, f2] += sa[i1] * sa[i2] * self.cl['bb', f1, f2]
                    dcl['bb', f1, f2] = (ca[i1] * ca[i2] * self.cl['bb', f1, f2] +
                                         sa[i1] * sa[i2] * self.cl['ee', f1, f2])
                if 'eb' in cls:
                    dcl['ee', f1, f2] += (-ca[i1] * sa[i2] * self.cl['eb', f1, f2] -
                                          sa[i1] * ca[i2] * self.cl['eb', f2, f1])
                    dcl['bb', f1, f2] += (sa[i1] * ca[i2] * self.cl['eb', f1, f2] +
                                          sa[i2] * ca[i1] * self.cl['eb', f2, f1])
                    dcl['eb', f1, f2] = (ca[i1] * sa[i2] * self.cl['ee', f1, f2] -
                                         sa[i1] * ca[i2] * self.cl['bb', f1, f2] +
                                         ca[i1] * ca[i2] * self.cl['eb', f1, f2] -
                                         sa[i1] * sa[i2] * self.cl['eb', f2, f1])
                if 'tb' in cls:
                    dcl['te', f1, f2] += -sa[i2] * self.cl['tb', f1, f2]
                    dcl['tb', f1, f2] = (sa[i2] * self.cl['te', f1, f2] +
                                         ca[i2] * self.cl['tb', f1, f2])

        return dcl
