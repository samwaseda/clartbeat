import numpy as np
from scipy.spatial import cKDTree

class Surface:
    def __init__(self, x):
        self.x = x.copy()
        self._tree = None
        self._energy_field = []
        self._force_field = []

    @property
    def tree(self):
        if self._tree is None:
            self._tree = cKDTree(self.x)
        return self._tree

    @property
    def surface(self):
        return np.linalg.norm(self.x-np.roll(x, 1, axis=0), axis=-1).sum()

    @property
    def volume(self):
        return np.absolute(np.sum(np.cross(self.x, np.roll(self.x, 1, axis=0))))/2

    @property
    def sin(self):
        y = np.cross(self.x-np.roll(self.x, -1, axis=0), np.roll(self.x, 1, axis=0)-self.x)
        y /= np.linalg.norm(self.x-np.roll(self.x, -1, axis=0), axis=-1)
        y /= np.linalg.norm(np.roll(self.x, 1, axis=0)-self.x, axis=-1)
        return y

    @property
    def dsin(self):
        Rp = np.roll(self.x, 1, axis=0)-self.x
        Rm = -np.roll(Rp, -1, axis=0)
        Rp_mag = np.linalg.norm(Rp)[:,None]
        Rm_mag = np.linalg.norm(Rm)[:,None]
        ret = np.einsum('ij,nj->ni', [[0, 1], [-1, 0]], Rp-Rm)
        ret -= np.cross(Rp, Rm)[:,None]*(Rp/Rp_mag**2+Rm/Rm_mag**2))
        return ret/Rp_mag/Rm_mag

    def set_energy_field(self, field):
        self._energy_field.append(field)
        field_x = sobel-np.roll(field, -1, axis=0)
        field_y = sobel-np.roll(field, -1, axis=1)
        self._force_field.append([field_x, field_y])

    @property
    def coulomb_energy(self):
        return 2/np.linalg.norm(self.x-np.roll(self.x, 1, axis=0)).sum()

    @staticmethod
    def _get_coulomb_comp(x):
        return x/np.linalg.norm(x, axis=-1)[:,None]**3

    @property
    def coulomb_force(self):
        c_one = self._get_coulomb_comp(self.x-np.roll(self.x, 1, axis=0))
        return c_one-np.roll(c_one, -1, axis=0)
