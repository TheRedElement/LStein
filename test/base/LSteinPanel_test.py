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
            (LSP1, (3.3, 11.0))
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
        
"""
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

        assert np.all(LSP.LSC.xlims[0] < x_cut)
        assert np.all(x_cut < LSP.LSC.xlims[1])
        assert np.all(LSP.ylims[0] < y_cut)
        assert np.all(y_cut < LSP.ylims[1])

class Test_project_xy_theta:

    @pytest.fixture(
        params=[
            (LSP1,
             np.linspace(0,15,9), np.linspace(0.2,8.0,9),
             np.array([0.17362002, 0.53449428, 1.06854129, 1.76028982, 2.59024175, 3.53711985, 4.5798561 , 5.69902841, 6.87768006]),
             np.array([-2.59419662, -3.8758189, -5.11457178, -6.29605321, -7.41034059, -8.45222985, -9.42075067, -10.3182475 , -11.1493281]),
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
             np.array([0.16676853, 0.53972186, 1.10641683, 1.86685343, 2.82103167, 3.96895154, 5.31061305, 6.8460162 , 8.57516098]),
             np.array([-2.6226378 , -3.88518615, -5.10655345, -6.28673969, -7.42574488, -8.52356901, -9.58021208, -10.5956741 , -11.56995506]),
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

"""