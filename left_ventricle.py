import numpy as np
from scipy.optimize import minimize
from scipy import ndimage
from surface import Surface

class LeftVentricle:
    def __init__(
        self,
        ref_job,
        sigma_l_to_r=None,
        sine_sigma=6,
        sine_weight=100,
        k_ratio=0.5,
        x_opt_lr=0.8,
        left_to_right_ratio=[0.75, 0.25]
    ):
        self.ref_job = ref_job
        self._er = None
        self._frame = None
        self._curvature = None
        self._lr_positions = None
        self._sigma_l_to_r = sigma_l_to_r
        self.sine_sigma = 6
        self.sine_weight = 10
        self._left_to_right = None
        self.k_ratio = k_ratio
        self.x_opt_lr = x_opt_lr
        self._convex_weight = None
        self.left_to_right_ratio = left_to_right_ratio
        self._perimeter = None

    def get_edges(self, max_dist=10):
        edges = []
        ep = np.array([self.er[1], -self.er[0]])
        for i, t in enumerate([self.ref_job.right, self.ref_job.left]):
            x = t.points.copy().astype(float)
            x -= self.ref_job.heart.get_center()
            y = np.einsum('ij,nj->ni', np.stack((self.er, ep)), x)
            cond = np.absolute(y[:,1])<max_dist
            if np.sum(cond)==0:
                return np.zeros(2)
            if i==0:
                edges.append(y[cond, 0].min())
            else:
                edges.append(y[cond, 0].max())
        return edges

    @property
    def er(self):
        if self._er is None:
            self._er = self.ref_job.right.get_center()-self.center
        return self._er/np.linalg.norm(self._er)

    def get_optimum_radius(self):
        def error_f(r, er=self.er, center=self.center, points=self.ref_job.right.points):
            return np.sum(np.absolute(np.linalg.norm(points-(r[0]*er+self.center), axis=-1)-r[1]))
        opt = minimize(error_f, [1, 100], method='Nelder-Mead')
        return opt

    @property
    def tot_perim(self):
        return self.ref_job.heart.perimeter.x

    @property
    def _rel_perim(self):
        return self.tot_perim-self.center

    @property
    def center(self):
        return self.ref_job.heart.get_center()

    @property
    def _center_to_edge(self):
        return np.max(np.einsum('i,ni->n', self.er, self.tot_perim-self.center))

    @property
    def _edge_to_right(self):
        return self._center_to_edge-np.linalg.norm(
            self.center-self.ref_job.right.get_center(ref_point=self.center)
        )

    @property
    def left_to_right(self):
        if self._left_to_right is None:
            self._left_to_right = np.sort(self.get_edges())
        return self._left_to_right 

    @property
    def x_opt(self):
        x_t = self._center_to_edge-2*self._edge_to_right
        x_m = np.dot([1-self.x_opt_lr, self.x_opt_lr], self.left_to_right)
        return (x_m*(1-self.k_ratio)+self.k_ratio*x_t)*self.er+self.center

    @property
    def curvature(self):
        if self._curvature is None:
            x = self._rel_perim
            y = np.cross(x-np.roll(x, -1, axis=0), np.roll(x, 1, axis=0)-x)
            y /= np.linalg.norm(x-np.roll(x, -1, axis=0), axis=-1)
            y /= np.linalg.norm(np.roll(x, 1, axis=0)-x, axis=-1)
            self._curvature = y
        return self._curvature

    @property
    def lv_end_args(self):
        p = np.cross(self.er, self._rel_perim)
        p_pos = p > 0
        p_neg = p < 0
        weight = self.weight
        args = np.sort([np.argmax(p_pos*weight), np.argmax(p_neg*weight)])
        if args[0] == args[1]:
            raise ValueError('Left ventricle edges not detected')
        return args

    @property
    def lv_end(self):
        if self._lr_positions is None:
            self._lr_positions = self._rel_perim[self.lv_end_args]
        return  self._lr_positions

    @property
    def sigma_l_to_r(self):
        if self._sigma_l_to_r is None:
            return np.ptp(self.left_to_right)/2
        return self._sigma_l_to_r

    @property
    def parallel_distance(self):
        return np.einsum('ni,i->n', self._rel_perim, self.er)

    @property
    def l_to_r_weight(self):
        dist = self.parallel_distance
        dist -= np.dot(self.left_to_right_ratio, self.left_to_right)
        return np.exp(-0.5*dist**2/self.sigma_l_to_r**2)

    @property
    def edge_to_r_weight(self):
        dist = self.parallel_distance-np.max(self.left_to_right)
        return dist < 0

    @property
    def weight(self):
        return self.convex_weight*self.l_to_r_weight*self.edge_to_r_weight

    @property
    def convex_weight(self):
        if self._convex_weight is None:
            self._convex_weight = np.log(
                1+np.exp(self.sine_weight*ndimage.gaussian_filter1d(
                    self.curvature, sigma=self.sine_sigma
                ))
            )**2
        return self._convex_weight

    @property
    def frame(self):
        if self._frame is None:
            ex = self.lv_end[1]-self.lv_end[0]
            if np.linalg.norm(ex) < 1:
                raise ValueError('Frame not recognized')
            ex /= np.linalg.norm(ex)
            ey = np.einsum('ij,j->i', [[0, 1], [-1, 0]], ex)
            ey *= np.sign(np.sum(np.mean(self.lv_end, axis=0)*ey))
            self._frame = np.stack((ex, ey))
        return self._frame

    @property
    def angle(self):
        angle = np.arcsin(self.curvature[self.lv_end_args[0]: self.lv_end_args[0]]).sum()
        angle -= np.pi*np.rint(angle/(2*np.pi))
        return angle

    @property
    def contact_area(self):
        return 0.5*np.absolute(np.sum(self.frame[0]*np.diff(self.lv_end, axis=0)))

    @property
    def center_to_end_vertical(self):
        return np.sum(self.frame[1]*self.lv_end[0])

    @property
    def new_center(self):
        mean_position = np.mean(self.lv_end, axis=0)
        vector = -self.frame[1]*self.center_to_end_vertical
        return mean_position+vector

    @property
    def new_radius(self):
        return np.linalg.norm(self.lv_end[0]-self.new_center)

    @property
    def epsilon(self):
        x = np.einsum('i,ji->j', self.x_opt-self.new_center-self.center, self.frame)
        return (1-x[1]/np.sqrt(self.new_radius**2-x[0]**2))/np.sqrt(1-(x[0]/self.contact_area)**2)

    def separate_points(self, x_input):
        x = np.atleast_2d(x_input)-self.new_center-self.center
        x = np.einsum('ij,nj->ni', self.frame, x)
        in_lv = np.array(len(x)*[True])
        cond_in = np.absolute(x[:,0]/self.contact_area)<1
        r = np.sqrt(self.new_radius**2-x[cond_in,0]**2)
        dr = (1-self.epsilon*np.sqrt(np.absolute(1-(x[cond_in,0]/self.contact_area)**2)))
        in_lv[cond_in] = r*dr<x[cond_in,1]
        return np.squeeze(in_lv)

    def get_left_ventricle(self):
        p_rel = self._rel_perim-self.new_center
        c = np.sum(p_rel*self.frame[1], axis=-1) >= self.center_to_end_vertical
        y = np.sum(self.frame[1]*p_rel[c], axis=-1)
        x = np.sum(self.frame[0]*p_rel[c], axis=-1)
        r = np.sqrt(self.new_radius**2-x**2)
        dr = (1-self.epsilon*np.sqrt(np.absolute(1-(x/self.contact_area)**2)))
        y -= r*dr
        p_rel[c] -= y[:,None]*self.frame[1]
        return p_rel+self.new_center+self.center

    @property
    def perimeter(self):
        if self._perimeter is None:
            self._perimeter = Surface(self.get_left_ventricle())
        return self._perimeter

