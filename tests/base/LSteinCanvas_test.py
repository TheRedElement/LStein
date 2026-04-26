
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
#%%tests
class Test_compute_xaxis:
    
    @pytest.fixture(
        params=[
            (LSC1, )
        ]
    )
    def action(self, request):
        #arrange
        
        #act
        LSC,  = request.param
        xaxis_specs = LSC.compute_xaxis()

        return xaxis_specs, LSC

    #assert
    def test_compute_ticks(self, action):
        circles_x, circles_y = action[0][0:2]
        LSC = action[1]
        
        assert len(LSC.xticks[0]) == len(circles_x)
        assert len(LSC.xticks[0]) == len(circles_y)

    def test_compute_ticklabs(self, action):
        xtickpos_x, xtickpos_y, xticklabs = action[0][2:5]
        LSC = action[1]

        assert len(LSC.xticks[0]) == len(xtickpos_x)
        assert len(LSC.xticks[0]) == len(xtickpos_y)
        assert len(LSC.xticks[0]) == len(xticklabs)

    def test_compute_labs(self, action):
        xlabpos_x, xlabpos_y = action[0][5:7]
        LSC = action[1]

        assert isinstance(xlabpos_x, (float,int))
        assert isinstance(xlabpos_y, (float,int))
         
class Test_compute_thetaaxis:
    
    @pytest.fixture(
        params=[
            (LSC1, )
        ]
    )
    def action(self, request):
        #arrange
        
        #act
        LSC,  = request.param
        thaxis_specs = LSC.compute_thetaaxis()

        return thaxis_specs, LSC

    #assert
    def test_compute_ticks(self, action):
        thetatickpos_xi, thetatickpos_yi, \
            thetatickpos_xo, thetatickpos_yo = action[0][0:4]
        LSC = action[1]

        ri = np.round(thetatickpos_xi**2 + thetatickpos_yi**2, 3)
        ro = np.round(thetatickpos_xo**2 + thetatickpos_yo**2, 3)
        
        assert len(LSC.thetaticks[0]) == len(thetatickpos_xi)
        assert len(LSC.thetaticks[0]) == len(thetatickpos_yi)
        assert len(LSC.thetaticks[0]) == len(thetatickpos_xo)
        assert len(LSC.thetaticks[0]) == len(thetatickpos_yo)
        assert len(np.unique(ri))==1
        assert len(np.unique(ro))==1
        assert np.all(ri < ro)

    def test_compute_ticklabs(self, action):
        thetaticklabelpos_x, thetaticklabelpos_y, thetaticklabs = action[0][4:7]
        LSC = action[1]
        assert len(LSC.thetaticks[0]) == len(thetaticklabelpos_x)
        assert len(LSC.thetaticks[0]) == len(thetaticklabelpos_y)
        assert len(LSC.thetaticks[0]) == len(thetaticklabs)

    def test_compute_labs(self, action):
        thlabpos_x, thlabpos_y = action[0][7:9]
        LSC = action[1]

        assert isinstance(thlabpos_x, (float,int))
        assert isinstance(thlabpos_y, (float,int))
         
    def test_compute_indicator(self, action):
        x_arrow, y_arrow = action[0][9:11]
        LSC = action[1]
        # print(len(x_arrow), len(y_arrow))

        assert True

class Test_compute_ylabel:
    
    @pytest.fixture(
        params=[
            (LSC1, )
        ]
    )
    def action(self, request):
        #arrange
        
        #act
        LSC,  = request.param
        thaxis_specs = LSC.compute_ylabel()

        return thaxis_specs, LSC

    #assert
    def test_compute_labs(self, action):
        ylabpos_x, ylabpos_y = action[0][0:2]
        LSC = action[1]

        assert isinstance(ylabpos_x, (float,int))
        assert isinstance(ylabpos_y, (float,int))
         
        assert True

