# Brief description of the example datasets

Demonstrates all supported file formats: csv, xlsx, txt, tsv

## `synth_signals.xlsx` / `synth_signals.csv`
Collection of synthetic signals showcasing pyBOATs capabilities
- signal1: deterministic chirp with very slow low amplitude trend and exponential decay
- signal2: deterministic trend dynamics much closer to signal, needs sinc fine tuning
- signal3: highly autocorrelated AR1 trend, also needs sinc fine tuning
- signal4: the former signal6, showing slowdown followed by speedup of frequency

## `FIJI_measurement.txt`
- real measurements of PSM dynamics, courtesy of Aulehla Lab @embl

## `AR1_0.7_ensemble.csv`
- completely aperiodic dataset
- 100 AR(1) realizations with $\alpha=0.7$
- ensemble dynamics reveal continuous background spectrum
- comma separated values (`csv`)

## `phase_diff_ensemble.tsv`
- ensemble of phase diffusing signals
- instantaneous periods fluctuations give rise to freely drifting phases
- show exponential decay of order parameter (batch process  -> ensemble dynamics)

##### Example analysis output: ensemble and time averaged Wavelet spectra
<img src="../doc/assets/AR1_globalFourier.png" alt="AR1 global Fourier spectrum" width="350"/>

Reproduce via:
- `Open` --> AR1_0.7_ensemble.csv
- click on arbitrary signal to trigger the dynamics defaults
- `Analyze all..`
- check the `Global Fourier Estimate` checkbox
- `Analyze 100 Signals!`
