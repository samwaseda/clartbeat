import numpy as np

class Surface:
    def __init__(self, x):
        self._x = x.copy()

    @property
    def x(self):
        return self._x

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
