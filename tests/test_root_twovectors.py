from functools import partial

import pytest
from pytest import approx
import numpy as np
from math import sqrt
assert_allclose = partial(np.testing.assert_allclose, atol=.0000001)

from hepvector import Vector2D
import itertools

ROOT = pytest.importorskip('ROOT')

def to_lv(rvec):
    return Vector2D(rvec.X(), rvec.Y())

def assert_same(lvec, rvec):
    assert_allclose(lvec, to_lv(rvec), atol=.0000001)

origvals = (
        (1, 1),
        (.5, .5),
        (0, 0),
        (12.1, 32.1),
        (.1, .3),
        (3.2, 1.3),
        (0, 32.1),
        (.1, 0)
        )

# Expand the above by every possible combination of minus signs (set to avoid duplicates)
negate = [item for i in range(2) for item in itertools.combinations(range(2),i)]
xyvals = set(tuple(i[n] if n not in j else -i[n] for n in range(2)) for i in origvals for j in negate)


@pytest.mark.parametrize("xy", xyvals)
def test_values(xy):
    np.seterr(divide='ignore', invalid='ignore')
    x,y = xy

    rvec = ROOT.TVector2(x,y)
    svec = Vector2D(x,y)

    assert_same(svec, rvec)
    # assert svec.cos_theta() == approx(rvec.CosTheta()) # Not implemented


    # Adding all the def's in LorentzVectors here, along with tests if applicable

    # def origin(cls):
    # def from_pandas(cls, pd_dataframe):
    # def from_vector(cls, other):
    # def dot(self, other):

    # def mag(self):
    assert svec.mag == approx(sqrt(rvec.Mod2()))

    # def mag2(self):
    assert svec.mag2 == approx(rvec.Mod2())

    # def unit(self, inplace=False):
    # def T(self):
    # def T(self, val):
    # def to_pd(self):
    # def angle(self, other, normal=None):
    # def __array_finalize__(self, obj):
    # def __array_wrap__(self, out_arr, context=None):
    # def __getitem__(self, item):
    # def __setitem__(self, item, value):
    # def dims(self):
    # def _repr_html_(self):


