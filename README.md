# SYSLibrary

Library of systematics template for SO likelihood analysis.

Main modules are
* `syslib.py`: master module with abstract classes
* `syslib_zzz.py`: zzz-specific extension of systematics template, with zzz beign an SO-derived probe

Currently available:
* `syslib_mflike.py`: collection of templates of systematics for the high-ell likelihood analysis

Currently available abstract classes are
* calibration
* additive systematics template from yaml file
* act on a dictionary of theory Cls

Currently available specific classes are:
* alm-based calibration (mflike)
* rotation of Cls due to miscalibration of polarization angle
* T-to-E leakage (Planck-based)

To get started, have a look at the notebooks in `sysspectra/notebook`

## Contributing
Current contributors to syslibrary are: Martina Gerbino, Luca Pagano. Special thanks to FGSpectra contributors (Davide Poletti, Max Abitbol, Zack Li). The first version of syslib was heavily based on fgspectra. Feel free to join: contributors are welcome!

## Dependencies
* Python > 3
* numpy / scipy
* yaml

## Installing
Since we're still putting this together, just install in developer mode for now.

```
pip install -e .
```

