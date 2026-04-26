#%%imports
import pytest
import numpy as np

from lstein import lstein

#%%global vars
LSC1 = lstein.LSteinCanvas(
    thetaticks=[10,20,40],
    yticks=[0,1,5],
    xticks=[1, 5, 12],
    thetaguidelims=(-np.pi/2,np.pi/2), xlimdeadzone=0.3
)
LSP1 = LSC1.add_panel(
    theta=12,
    panelsize=np.pi/10,
)

#%%tests
class Test_get_thetabounds:

    @pytest.fixture(
        params=[
            (LSP1, (-1.1912, -1.3483, -1.0341))
        ]
    )
    def action(self, request):
        #arrange
        
        #act
        LSP, truth  = request.param
        thbounds = LSP.get_thetabounds()

        return thbounds, LSP, truth

    #assert
    def test_output(self, action):
        theta_offset, theta_lb, theta_ub = action[0][0:3]
        theta_offset_tr, theta_lb_tr, theta_ub_tr = action[2][0:3]
        assert theta_offset == pytest.approx(theta_offset_tr, rel=1e-3)
        assert theta_lb == pytest.approx(theta_lb_tr, rel=1e-3)
        assert theta_ub == pytest.approx(theta_ub_tr, rel=1e-3)
        
class Test_get_rbounds:

    @pytest.fixture(
        params=[
            (LSP1, (0.3, 1.0))
        ]
    )
    def action(self, request):
        #arrange
        
        #act
        LSP, truth  = request.param
        rbounds = LSP.get_rbounds()

        return rbounds, LSP, truth

    #assert
    def test_output(self, action):
        r_lb, r_ub = action[0][0:2]
        r_lb_tr, r_ub_tr = action[2][0:2]
        assert r_lb == pytest.approx(r_lb_tr, rel=1e-3)
        assert r_ub == pytest.approx(r_ub_tr, rel=1e-3)
        
class Test_get_yticks:
    
    @pytest.fixture(
        params=[
            (LSP1,)
        ]
    )
    def action(self, request):
        #arrange
        
        #act
        LSP,  = request.param
        _, th_lb, th_ub = LSP.get_thetabounds()
        pred = LSP.get_yticks(th_lb, th_ub)

        return pred, LSP

    #assert
    def test_get_yticks(self, action):
        ytickpos_th, yticklabs = action[0][0:2]
        LSP = action[1]
        
        assert len(LSP.yticks[0]) == len(ytickpos_th)
        assert len(LSP.yticks[0]) == len(yticklabs)

class Test_apply_axis_limits:

    @pytest.fixture(
        params=[
            (LSP1, np.linspace(0,15,9), np.linspace(0.2,8.0,9))
        ]
    )
    def action(self, request):
        #arrange
        
        #act
        LSP, x, y  = request.param
        pred = LSP.apply_axis_limits(x, y)

        return pred, LSP

    #assert
    def test_output(self, action):
        x_cut, y_cut, kwargs = action[0][0:3]
        LSP = action[1]

        assert np.all(LSP.LSC.xlims_data[0] < x_cut)
        assert np.all(x_cut < LSP.LSC.xlims_data[1])
        assert np.all(LSP.ylims_data[0] < y_cut)
        assert np.all(y_cut < LSP.ylims_data[1])

class Test_project_xy_theta:

    @pytest.fixture(
        params=[
            (LSP1,
                np.linspace(0,15,9), np.linspace(0.2,8.0,9),
                np.array([0.05584462, 0.1101468 , 0.17837681, 0.25634874, 0.34041605, 0.42801352, 0.51754656, 0.60808895, 0.69912152]),
                np.array([-0.22967182, -0.33819704, -0.44023484, -0.53618973, -0.62721111, -0.71457519, -0.7993553 , -0.88234625, -0.96410247])             
            )
        ]
    )
    def action(self, request):
        #arrange
        
        #act
        LSP, x, y, x_proj_tr, y_proj_tr  = request.param
        pred = LSP.project_xy_theta(x, y)

        return pred, LSP, (x_proj_tr, y_proj_tr)

    #assert
    def test_output(self, action):
        x_proj, y_proj = action[0][0:3]
        LSP = action[1]
        x_proj_tr, y_proj_tr = action[2][0:3]
        assert np.all(x_proj == pytest.approx(x_proj_tr, rel=1e-3))
        assert np.all(y_proj == pytest.approx(y_proj_tr, rel=1e-3))

class Test_project_xy_y:

    @pytest.fixture(
        params=[
            (LSP1,
                np.linspace(0,15,9), np.linspace(0.2,8.0,9),
                np.array([0.05531838, 0.10817095, 0.17774801, 0.26404954, 0.36707555, 0.48682604, 0.623301  , 0.77650045, 0.94642436]),
                np.array([-0.2324104 , -0.33978792, -0.44049305, -0.53452579, -0.62188614, -0.7025741 , -0.77658967, -0.84393285, -0.90460364]),
            )
        ]
    )
    def action(self, request):
        #arrange
        
        #act
        LSP, x, y, x_proj_tr, y_proj_tr  = request.param
        pred = LSP.project_xy_y(x, y)

        return pred, LSP, (x_proj_tr, y_proj_tr)

    #assert
    def test_output(self, action):
        x_proj, y_proj = action[0][0:3]
        LSP = action[1]
        x_proj_tr, y_proj_tr = action[2][0:3]
        assert np.all(x_proj == pytest.approx(x_proj_tr, rel=1e-3))
        assert np.all(y_proj == pytest.approx(y_proj_tr, rel=1e-3))
