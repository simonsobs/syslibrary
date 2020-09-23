# SYSspectra

Library for the evaluation of the systematics template for SO hi-ell likelihood analysis.

Main modules are
* `syslib.py`: evaluation of systematics in ell-space

Currently available features are
* calibration
* additive systematics template from yaml file
* T-to-E beam leakage

To get started, have a look at the notebooks in `sysspectra/notebook`

## Contributing
Current contributors to SYSspectra are: Martina Gerbino. Special thanks to FGSpectra contributors (Davide Poletti, Max Abitbol, Zack Li). The first version of syslib was heavily based on fgspectra. Feel free to join: contributors are welcome!

## Dependencies
* Python > 3
* numpy / scipy
* yaml

## Installing
Since we're still putting this together, just install in developer mode for now.

```
pip install -e .
```

