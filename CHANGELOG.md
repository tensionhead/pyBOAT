
#### pyBOAT 0.9.4

- just added the link to the gitter chat to the Help menu
- removed bulky talk pdf

#### pyBOAT 0.9.3

- fixed UI bugs concerning ridge smoothing/thresholding and SSG noise strength

#### pyBOAT 0.9.2

- Scripting interface for significance tests with empirical backgrounds,
see the new [empirical_backgrounds_demo.py](empirical_backgrounds_demo.py)
- Added new output settings to control number and graphics formats
- Batch processing allows to export the filtered signals
- New normalization for the average signal power

### pyBOAT 0.9

- New Settings widget to store persistent parameter values
- Reworked the API, see the new `scripting_demo.py` in the repo root
- Fixed amplitude envelope plotting issues
- Added a lot more documentation for the API functions
- Fixed one-column txt file parsing error
- capped sinc filter size to 2000 for performance reasons
- No automatic overwrite of analyzer parameters for the synthetic generator
- new taskbar icon :boat:

### pyBOAT 0.8.22

- Fixed export paths for Windows platforms
- Fixed synthetic signal generator AR(1) alpha setting
- Added batch export of wavelet spectra w/o ridges

### pyBOAT 0.8.20

- Added time averaging of Wavelet spectra <-> Fourier estimates
- Added Fourier distribution for ensembles, can be used for empirical background estimation
- Reworked FFT visualizations
- Status bars with tool tips for all analysis windows

### pyBOAT 0.8.17

- Fixed crashes during batch analysis with thresholded ridges
- Warning and on-the-fly interpolation of non-contiguous missing values (NaNs)

### pyBOAT 0.8.16

- added the pyBOAT icon
- changed version numbering scheme

### pyBOAT 0.8.1.5

- fixed Fourier spectra crash
- changed plotting style for short signals

### pyBOAT 0.8.1

- fixed COI on thresholded ridge
- improved summary statistics

### pyBOAT 0.8

- added batch processing menu
- summary statistics: time-averaged power and ensemble dynamics

### pyBOAT 0.7.6.1

- reworked the SSG, catching missing parameter crashes

### pyBOAT 0.7.6

- Added general data import options with optional missing value interpolation
- Added NaN interpolation to the core module

### pyBOAT 0.7.5

- Synthetic Signal Generator, yeah!

### pyBOAT 0.7.4

- Amplitude envelope estimation and removal

	
