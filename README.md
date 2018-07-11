![HEPVector](images/HEPvector.png)

[![image](https://img.shields.io/pypi/v/hepvector.svg)](https://pypi.python.org/pypi/hepvector)
[![image](https://travis-ci.com/henryiii/hepvector.svg?branch=master)](https://travis-ci.com/henryiii/hepvector)
[![Documentation Status](https://readthedocs.org/projects/hepvector/badge/?version=latest)](https://hepvector.readthedocs.io/en/latest/?badge=latest)
[![Updates](https://pyup.io/repos/github/henryiii/hepvector/shield.svg)](https://pyup.io/repos/github/henryiii/hepvector/)

Numpy based vectors for general purpose calculation and physics. Designed for the SciKit-HEP project, but can be used stand-alone with no dependencies except Numpy (extra performance can be obtained if Numba is present).

## Features

The key design feature is array support. While non-array is also possible (backward compatible), arrays are many times faster and utilize all of Numpy's speed. An example of an array:
  ```python
  v = Vector3D(1, [2,3], 4)
  ```
  would make two vectors, [1,2,4] and [1,3,4] (standard Numpy casting rules). In fact, the vector classes are simply subclasses of NDArrays, so anything that works with arrays works with Vectors, and Vectors can be cast to/from Numpy without a copy.
* Optional speedups with Numba for a few parts (more eventually)
* Phase space generator, similar speed (within a factor of 2 or so) to compiled ROOT, but fully in Python! Even works on an iPad. Also makes a good example of what HEPvector code looks like.
* Lorentz, 3D, and 2D vectors, and all common operations in a single vector base class. More metrics and dimensions can be added by users.
* Free software: BSD license
<!--   Documentation: <https://hepvector.readthedocs.io>. -->

## Performance

Generation of phase space:
* 1 at a time: 5.18 ± 0.06 ms each
* 1,000,000 at a time: 0.80 ± 0.04 µs each (803 ms total)

So that’s 6,000 times faster if you generate 1,000,000 events using arrays instead of placing event-per-event code in loops.

## Usage

You can create a vector several ways:

```python
v = LorentzVector(x,y,z,t) # can be values or arrays
v = arr.view(LorentzVector) # Convert an existing array
v = LorentzVector.from_vector(arr) # Can also copy
v = LorentzVector.from_pandas(df) # DataFrames, too
v = Vector3D.from_spherical_coords(r, theta, phi)
v = Vector3D.from_cylindrical_coords(rho, phi, z)

```

You can use a variety of methods:

```python
dot_product = a.dot(b)
cross_product = a.dot(b)
boost_vec = a.boost()
````

(and others, see the docs.)

## Credits

HEPvector was created by Henry Schreiner.

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
