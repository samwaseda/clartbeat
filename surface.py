import numpy as np
from scipy.spatial import cKDTree

class Surface:
    def __init__(self, x):
        self.x = x.copy()
        self._tree = None
        self._energy_field = None
        self._force_field = None

    @property
    def xp(self):
        return np.roll(self.x, 1, axis=0)

    @property
    def xm(self):
        return np.roll(self.x, -1, axis=0)

    @property
    def Rp(self):
        return self.xp-self.x

    @property
    def Rm(self):
        return self.x-self.xm

    @property
    def rp(self):
        return self.Rp/np.linalg.norm(self.Rp, axis=-1)[:,None]

    @property
    def rm(self):
        return self.Rm/np.linalg.norm(self.Rm, axis=-1)[:,None]

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
        return np.cross(self.rp, self.rm)

    @property
    def dsin(self):
        Rp_mag = np.linalg.norm(self.Rp, axis=-1)[:,None]
        Rm_mag = np.linalg.norm(self.Rm, axis=-1)[:,None]
        ret = np.einsum('ij,nj->ni', [[0, 1], [-1, 0]], self.Rp-self.Rm)
        ret -= np.cross(self.Rp, self.Rm)[:,None]*(self.Rp/Rp_mag**2+self.Rm/Rm_mag**2)
        return ret/Rp_mag/Rm_mag

    @property
    def _indices(self):
        p = np.rint(self.x).astype(int)
        p[p<0] = 0
        p[p[:,0]>=self._energy_field.shape[0],0] = self._energy_field.shape[0]-1
        p[p[:,1]>=self._energy_field.shape[1],1] = self._energy_field.shape[1]-1
        return p

    @property
    def force_field(self):
        p = self._indices
        return np.stack((
            self._force_field[0, p[:,0], p[:,1]], self._force_field[1, p[:,0], p[:,1]]
        ), axis=-1)

    @property
    def energy_field(self):
        p = self._indices
        return np.sum(self._energy_field[p[:,0], p[:,1]])

    def set_energy_field(self, field):
        if self._energy_field is None:
            self._energy_field = field
        else:
            self._energy_field += field
        field_x = field-np.roll(field, -1, axis=0)
        field_y = field-np.roll(field, -1, axis=1)
        if self._force_field is None:
            self._force_field = np.array([field_x, field_y])
        else:
            self._force_field += np.array([field_x, field_y])

    @property
    def coulomb(self):
        return 2/np.linalg.norm(self.x-np.roll(self.x, 1, axis=0)).sum()

    @staticmethod
    def _get_coulomb_comp(x):
        return x/np.linalg.norm(x, axis=-1)[:,None]**3

    @property
    def dcoulomb(self):
        c_one = self._get_coulomb_comp(self.Rp)
        return c_one-np.roll(c_one, -1, axis=0)

    @property
    def hook(self):
        return np.sum(np.linalg.norm(self.Rp, axis=-1))

    @property
    def dhook(self):
        return -self.Rp+self.Rm

