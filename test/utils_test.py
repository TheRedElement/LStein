#%%imports
import pytest
import matplotlib.colors as mcolors
import numpy as np

from lstein import lstein
from lstein import utils as lsu

#%%tests
class Test_carth2polar:

    @pytest.fixture(
        params=[
            ( 0,  0, 0,          np.pi,),
            ( 1,  1, np.sqrt(2), np.pi/4+np.pi,),
            (-1, -1, np.sqrt(2), -3*np.pi/4+np.pi,),
            ( 1,  0, 1,          0+np.pi,),
            ( 0,  1, 1,          np.pi/2+np.pi,),
            (-1,  0, 1,          np.pi+np.pi,),
            ( 0, -1, 1,          -np.pi/2+np.pi,),
        ]
    )
    def action(self, request):
        #arrange
        
        #act
        x, y, r, th  = request.param
        pred = lsu.carth2polar(x, y)

        return pred, (r, th)

    #assert
    def test_output(self, action):
        r, th = action[0]
        r_tr, th_tr = action[1]
        
        assert r== pytest.approx(r_tr, rel=1e-3)
        assert th == pytest.approx(th_tr, rel=1e-3)

class Test_polar2carth:

    @pytest.fixture(
        params=[
            (0,          np.pi,      0,  0, ),
            (np.sqrt(2), np.pi/4,    1,  1, ),
            (np.sqrt(2), -3*np.pi/4,-1, -1, ),
            (1,          0,          1,  0, ),
            (1,          np.pi/2,    0,  1, ),
            (1,          np.pi,     -1,  0, ),
            (1,          -np.pi/2,   0, -1, ),
        ]
    )
    def action(self, request):
        #arrange
        
        #act
        r, th, x, y  = request.param
        pred = lsu.polar2carth(r, th)

        return pred, (x, y)

    #assert
    def test_output(self, action):
        x, y = action[0]
        x_tr, y_tr = action[1]
        
        assert np.round(x, 3) == pytest.approx(x_tr, rel=1e-3)
        assert np.round(y, 3) == pytest.approx(y_tr, rel=1e-3)

class Test_minmaxscale:

    @pytest.fixture(
        params=[
            (np.linspace(-20,20,3), -5, 5, -40, 40, np.array([-2.5, 0., 2.5])),
            (np.linspace(-20,20,3), -5, 5, None, None, np.array([-5.0, 0., 5.0])),
        ]
    )
    def action(self, request):
        #arrange
        
        #act
        x, xmin, xmax, xmin_ref, xmax_ref, x_tr  = request.param
        pred = lsu.minmaxscale(x, xmin, xmax, xmin_ref, xmax_ref)

        return pred, x_tr

    #assert
    def test_output(self, action):
        x = action[0]
        x_tr = action[1]
        assert np.all(x == pytest.approx(x_tr, rel=1e-3))


class Test_get_colors:

    @pytest.fixture(
        params=[
            (np.linspace(-20,20,3), "viridis", mcolors.Normalize, ['#440154', '#21918c', '#fde725']),
            (np.linspace(-20,20,5), "nipy_spectral", mcolors.Normalize, ['#000000', '#0078dd', '#00bc00', '#ffc900', '#cccccc']),
        ]
    )
    def action(self, request):
        #arrange
        
        #act
        x, cmap, norm, colors_tr = request.param
        colors = lsu.get_colors(x, cmap, norm)

        return colors, colors_tr

    #assert
    def test_output(self, action):
        colors = action[0]
        colors_tr = action[1]
        print(repr(colors))

        assert np.all(colors == colors_tr)

