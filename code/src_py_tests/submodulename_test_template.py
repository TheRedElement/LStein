
#%%imports
import pytest
import submodulename    #your module to test

#additional imports
from astropy.cosmology import FlatLambdaCDM

#%%tests
class TestClass:
    
    @pytest.fixture(    #will be executed before each test
        params=[
            ("p11","p12","p13",...),
            ("p21","p22","p23",...),
        ]
    )
    def action(self, request):
        #arrange
        #NOTE: some settings (ground truth, underlying model...)
        
        ##get parameters
        p1, p2, p3 = request.param

        #act
        #NOTE: execute the funciton to test
        o1, o2, o3 = function_to_test(p1, p2, p3)

        return o1, o2, o3

    #assert
    def test_t1(self, action):
        p1, p2, p3, ... = action
        assert True

    def test_t2(self, action):
        assert False


#alterntive using parametrize (calls function for each parameter combination)
@pytest.mark.parametrize(
    "p1, p2, p3",
    [
        ("p11","p12","p13",...),
        ("p21","p22","p23",...),
    ]
)
def test_t1(p1, p2, p3):
    assert True