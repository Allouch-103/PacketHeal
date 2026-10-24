"""Hand calculations for the independent past-index verification formulas."""

from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_phase2_report import direct_forecasts


class DirectFormulaTests(unittest.TestCase):
    def test_received_spacing_recovery_and_capped_signed_channels(self):
        truth=np.array([[0.,0],[4,-4],[99,99],[99,99],[10,-10],[99,99]])
        mask=np.array([False,False,True,True,False,True])
        values=direct_forecasts(truth,mask,[3,1])
        np.testing.assert_allclose(values['linear'],[[0,0],[4,-4],[7,-5],[10,-6],[10,-10],[12,-11]])
        np.testing.assert_array_equal(values['locf'],[[0,0],[4,-4],[4,-4],[4,-4],[10,-10],[10,-10]])
        np.testing.assert_array_equal(values['locf'],values['alpha_beta'])

    def test_startup_unavailable_then_single_received_fallback(self):
        truth=np.array([[100.],[5.],[100.]])
        values=direct_forecasts(truth,np.array([True,False,True]),[2])
        self.assertTrue(np.isnan(values['linear'][0,0]))
        np.testing.assert_array_equal(values['linear'][1:],[[5],[5]])


if __name__=='__main__':
    unittest.main()
