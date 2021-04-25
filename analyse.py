import numpy as np
from process import ProcessImage
from tissue import Tissue
from area import Area
from left_ventricle import LeftVentricle
import json

class Analyse:
    def __init__(self, file_name, parameters=None):
        if parameters is None:
            with open('default_parameters.txt', 'r') as f:
                parameters = json.load(f)
        self.parameters = parameters
        self.image = ProcessImage(file_name=file_name, parameters=self.parameters['processing'])
        self._heart = None
        self._right = None
        self._left = None
        self._right = None
        self._tissue = None
        self._left_ventricle = None

    @property
    def heart(self):
        if self._heart is None:
            self.image.run_cluster('heart', **self.parameters['heart'])
            self._heart = self.image.get_data('heart')
        return self._heart

    @property
    def left(self):
        if self._left is None:
            if self._heart is None:
                _ = self.heart
            self.image.run_cluster('white', **self.parameters['white'])
            self._left = self.image.get_data(
                'white', self.image.get_index('left', **self.parameters['left'])
            )
        return self._left

    @property
    def right(self):
        if self._right is None:
            if self._left is None:
                _ = self.left
            self._right = self.image.get_data(
                'white', self.image.get_index(
                    'right',
                    **self.parameters['right'],
                )
            )
        return self._right

    @property
    def tissue(self):
        if self._tissue is None:
            self._tissue = Tissue(
                img=self.get_image(),
                total_area=self.image.get_area('heart'),
                white_areas=self.image.cluster['white'],
                **self.parameters['tissue']
            )
        return self._tissue

    @property
    def left_ventricle(self):
        if self._left_ventricle is None:
            if len(self.left.points)==0:
                return None
            self._left_ventricle = LeftVentricle(self)
        return self._left_ventricle

    def get_base_color(self, mean=True):
        return self.image.get_base_color(mean=mean)

    def get_image(self, mean=False):
        return self.image.get_image(mean=mean)

