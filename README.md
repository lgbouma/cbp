# install

Get a working version of anaconda (e.g., `module load anaconda3` when on
PICSciE clusters). Once anaconda is on your path:
```bash
conda env create -f environment.yml
```

This requires up to date C and fortran compilers (e.g., `gcc`, `gfortran`).

Oh, also for bleeding-edge `astrobase` (by default `pip install` gives an old
version with bugs that have been fixed):
```bash
$ git clone https://github.com/waqasbhatti/astrobase
$ cd astrobase
$ python setup.py install
$ # or use pip install . to install requirements automatically
$ # or use pip install -e . to install in develop mode along with requirements
```
