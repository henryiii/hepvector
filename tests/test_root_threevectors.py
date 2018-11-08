from functools import partial

import pytest
from pytest import approx
import numpy as np
assert_allclose = partial(np.testing.assert_allclose, atol=.0000001)

from hepvector import Vector3D
import itertools

ROOT = pytest.importorskip('ROOT')

def to_lv(rvec):
    return Vector3D(rvec.X(), rvec.Y(), rvec.Z())

def assert_same(lvec, rvec):
    assert_allclose(lvec, to_lv(rvec), atol=.0000001)

origvals = (
        (1, 1, 1),
        (.5, .5, .5),
        (0, 0, 0),
        (12.1, 32.1, 2.2),
        (.1, .3, .2),
        (3.2, 1.3, .5),
        (0, 32.1, .43),
        (.1, .2, 2.1)
        )

# Expand the above by every possible combination of minus signs (set to avoid duplicates)
negate = [item for i in range(4) for item in itertools.combinations(range(0,3),i)]
xyzvals = set(tuple(i[n] if n not in j else -i[n] for n in range(3)) for i in origvals for j in negate)

# t values
values = [-1, -.5, 0., .1, .5, 1., 2.32]

# slow but covers a lot
# @pytest.mark.parametrize("x", values)
# @pytest.mark.parametrize("y", values)
# @pytest.mark.parametrize("z", values)


@pytest.mark.parametrize("xyz", xyzvals)
def test_values(xyz):
    np.seterr(divide='ignore', invalid='ignore')
    x,y,z = xyz

    rvec = ROOT.TVector3(x,y,z)
    svec = Vector3D(x,y,z)

    assert_same(svec, rvec)
    # assert svec.cos_theta() == approx(rvec.CosTheta()) # Not implemented


    # Adding all the def's in LorentzVectors here, along with tests if applicable

    # def origin(cls):
    # def from_pandas(cls, pd_dataframe):
    # def from_vector(cls, other):
    # def dot(self, other):

    # def mag(self):
    assert svec.mag == approx(rvec.Mag())

    # def mag2(self):
    assert svec.mag2 == approx(rvec.Mag2())

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
    # def __new__(cls, x=0, y=0, dtype=np.double):

    # def phi(self):
    assert svec.phi == approx(rvec.Phi())

    # def rho(self):
    # assert svec.rho == approx(rvec.Rho()) # Fails

    # def angle(self, other):
    # def pt2(self):
    # def pt(self):
    # def __new__(cls, x=0, y=0, z=0, dtype=np.double):
    # def cross(self, other):

    # def theta(self):
    assert svec.theta == approx(rvec.Theta())

    # def r(self):
    # def in_basis(self, xhat, yhat, zhat):
    # def rotate_axis(self, axis, angle):
    # def rotate_euler(self, phi=0, theta=0, psi=0):
    # def from_spherical_coords(cls, r, theta, phi):
    # def from_cylindrical_coords(cls, rho, phi, z):


