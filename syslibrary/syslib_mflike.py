import numpy as np
from .syslib import Systematic


class TtoEleak_Planck15(Systematic):
    r"""
    T2E leakage template a la Planck
    te'(nu1,nu2) = te(nu1,nu2)+enu(l,nu2)*tt(nu1,nu2)
    ee'(nu1,nu2) = ee(nu1,nu2)+enu(l,nu1)*te(nu1,nu2)+enu(l,nu2)*te(nu2,nu1)
                  +enu(l,nu1)*enu(l,nu2)*tt(nu1,nu2)
    enu(l,nu1)=enu0(nu1)+enu2(nu1)*l**2+enu4(nu1)*l**4
    """

    def get_delta_cl(self, cl: dict, enu={'100': [0., 0., 0.]}) -> dict:
        """
        :param enu: 
        :param cl: dictionary of cls. Must be in the following format:
               cl[spec,f1,f2], with e.g., cl=tt,te,ee,etc and f1,f2=freqs
        :return: delta cl dict
        """

        self.enul = dict()
        for k1, enu_arr in enu.items():
            self.enul[k1] = enu_arr[0] + enu_arr[1] * self.ell ** 2 + enu_arr[2] * self.ell ** 4
        dcl = dict()

        for c1, f1 in enumerate(self.freq):
            for f2 in self.freq[c1:]:
                dcl["tt", f1, f2] = np.zeros_like(cl["tt", f1, f2])
                dcl["tt", f2, f1] = np.zeros_like(cl["tt", f1, f2])
                dcl["te", f1, f2] = self.enul[f2] * cl["tt", f1, f2]
                dcl["ee", f1, f2] = self.enul[f2] * cl["te", f2, f1] + self.enul[f1] * cl["te", f1, f2] + \
                                    self.enul[f1] * self.enul[f2] * cl["tt", f1, f2]
                # dcl["ee",f1,f2] = self.elnu2*self.cl["te",f1,f2] + self.elnu1*self.cl["te",f1,f2] +self.elnu1*self.elnu2*self.cl["tt",f1,f2]
                dcl["te", f2, f1] = dcl["te", f1, f2]  # self.elnu1*self.cl["tt",f2,f1]
                dcl["ee", f2, f1] = dcl[
                    "ee", f1, f2]  # self.elnu1*self.cl["te",f1,f2] + self.elnu2*self.cl["te",f2,f1] +self.elnu1*self.elnu2*self.cl["tt",f2,f1]

        return dcl


class Rotation_alm(Systematic):
    r"""
    apply rotation of C_ell due to polangle miscalibration
    each freq channel nu is rotated by its own polangle alpha_nu
    NB: alpha must be in deg
    """

    def get_rotated_cl(self, cl: dict, alpha: list):
        """
        :param alpha: array of polangle miscalibration in degrees
        :param cl: dictionary of cls. Must be in the following format:
               cl[spec,f1,f2], with e.g., cl=tt,te,ee,etc and f1,f2=freqs
        """

        # NB must check that len(alpha)==len(freq)
        ang = np.deg2rad(2. * np.array(alpha))
        cos_ang = np.cos(ang)
        sing_ang = np.sin(ang)
        dcl = dict()

        for f1, ca1, sa1 in zip(self.freq, cos_ang, sing_ang):
            for f2, ca2, sa2 in zip(self.freq, cos_ang, sing_ang):

                dcl['te', f1, f2] = ca2 * cl['te', f1, f2]
                dcl['ee', f1, f2] = ca1 * ca2 * cl['ee', f1, f2]
                dcl['tt', f1, f2] = cl['tt', f1, f2]
                if 'bb' in self.requested_cls:
                    dcl['ee', f1, f2] += sa1 * sa2 * cl['bb', f1, f2]
                    dcl['bb', f1, f2] = (ca1 * ca2 * cl['bb', f1, f2] +
                                         sa1 * sa2 * cl['ee', f1, f2])
                if 'eb' in self.requested_cls:
                    dcl['ee', f1, f2] += (-ca1 * sa2 * cl['eb', f1, f2] -
                                          sa1 * ca2 * cl['eb', f2, f1])
                    dcl['bb', f1, f2] += (sa1 * ca2 * cl['eb', f1, f2] +
                                          sa2 * ca1 * cl['eb', f2, f1])
                    dcl['eb', f1, f2] = (ca1 * sa2 * cl['ee', f1, f2] -
                                         sa1 * ca2 * cl['bb', f1, f2] +
                                         ca1 * ca2 * cl['eb', f1, f2] -
                                         sa1 * sa2 * cl['eb', f2, f1])
                if 'tb' in self.requested_cls:
                    dcl['te', f1, f2] += -sa2 * cl['tb', f1, f2]
                    dcl['tb', f1, f2] = (sa2 * cl['te', f1, f2] +
                                         ca2 * cl['tb', f1, f2])

        return dcl
