## pyBOAT - The Biological Oscillations Analysis Toolkit ##


Tools for time-frequency analysis with Morlet Wavelets
Inspired by 'A Practical Guide to Wavelet Analysis' (Torrence
and Compo 1998), 'Identification of Chirps with Continuous Wavelet Transform'
(Carmona, Hwang and Torresani 1995)
and [The Scientist and Engineer's Guide to Digital Signal Processing](http://www.dspguide.com/).

Version 0.7 April 2020, Frieda Sorgenfrei and Gregor Mönke. 

Questions etc. please to gregor.moenke@embl.de.

**Contributers are welcome!**

### Features ###

* Optimal sinc filter
* Fourier analysis
* Wavelet analysis 
* Ridge detection
* Phase extraction
* Amplitude estimation

### Things not ready yet ###

* Synthetic signal generator

### Installation ###

pyBOAT is written in Python and therefore requires Python to be present
on the system.
An easy way to install a cross-platform scientfic Python
environment is to use the freely availabe [Anaconda](https://www.anaconda.com/).
Installation instructions can be found here: https://docs.anaconda.com/anaconda/install/

#### Using  pip ####

pyBOAT is listed on the [Python Package Index (PyPI)](https://pypi.org/project/pyboat)
and can be directly installed using ```pip```. In case you don't have/want Anaconda, 
see here for install instructions for pip itself: https://pip.pypa.io/en/stable/installing/. 

To install pyboat simply type

```pip install pyboat```

into the command line. This makes the ```pyboat``` Python module available for import.
The graphical user interface (GUI) can be started with typing

```
pyboat
```

into the command line.

#### Running the GUI from source ###

To run the program directly from this repository, Python and several of its core 
scientific libraries have to be installed. Details can be found in the
[pyproject.toml](./pyproject.toml) under [requirements].


##### Mac OS #####

After downloading the repository, double click the 
``` pyBOAT_MacOS.command ``` file. It will open a 
terminal in the background and runs the pyBOAT program.
You might have to 'allow' 3rd-party apps to run, this
can be done for **El Capitan** by:

``` System Preferences -> Security & Privacy -> Allow Apps downloaded from -> Anywhere ```

For the newest version **Sierra** do a right click on that file,
and choose open.

##### Linux #####

Just run ```python -m pyboat ``` on the terminal 
from the root directory of this repository.

##### Windows #####

Run ```python -m pyboat ``` on the Windows command line
inside the root directoy of this repository.

##### Anaconda troubleshooting #####

In case of errors from Anaconda, you can try to update
your installation by typing

```conda update --all ```

in the terminal.


### Usage ###
-------------

#### Data import ####

Just open your saved time-series data by using ``` Open ``` 
from the (small) main window. Supported input formats are:
``` .xls, .xlsx, .csv, .tsv and .txt ```. For other file
extensions, white space separation of the data is assumed.
Please see examples of the supported formats in the 
``` example_data ``` directory of this repository.

#### Analysis ####

After successful import, you can simply click on the table representing
your data to select a specific time-series in the ``` DataViewer ```. 
Alternatively, select a specific time-series from the drop-down menu in the upper left.
To get the correct numbers/units you can change the ```Sampling Interval```
and ```Time Unit``` name in the top line of the ``` DataViewer ```. 
The general layout of the ```DataViewer``` to set up the analysis is shown here:

![DataViewer overview][DataViewer]

[DataViewer]:./doc/DataViewer.png?raw=true


##### Detrending  #####


The featured sinc-detrending is an optimal high-pass filter and removes low frequencies (high periods) 
from the signal via a sharp ``` cut-off-period ```. Details of the implementation can be found at 
[The Scientist and Engineer's Guide to Digital Signal Processing](http://www.dspguide.com/).
Check the ``` Trend ``` and/or ``` Detrended Signal ``` checkbox(es) 
and click ``` Refresh Plot ``` 
to see the effect of the filter on the selected time series.

##### Amplitude Envelope #####

If there is a strong trend in the amplitudes alone, for example a slow decay, pyBOAT offers
a simple sliding-window operation to estimate this envelope. The ```Window Size```
controls the time-window in which each amplitude is estimated. 
Check the ``` Envelope ``` checkbox and click ``` Refresh Plot ``` 
to see the detected envelope. When running the
Wavelet analysis, there is an option ```Normalize with Envelope``` to remove it
from the signal.

##### Set up Wavelet Analysis #####

Set the parameters for the Wavelet Analysis in the lower right:

| Input Field   | Meaning    |
| --- | --- |
| Smallest Period | Lower period bound <br> (Nyquist limit built-in)  |
| Number of Periods | Resolution of the transform <br> or number of convolutions             |   
| Highest Period | Upper period bound <br> Should not exceed observation time     |
| Expected maximal power | Upper bound for the colormap <br> indicating the Wavelet power <br> normalized to white noise |

Leave the ``` Use the detrended signal ``` box checked if you want to use the sinc-detrending. 
``` Analyze Signal ``` will perform the Wavelet transform of the selected signal. 

#### Wavelet Power Spectrum  ####

The input signal for the Wavelet analysis and the resulting 2d-power-spectrum are shown with aligned time axis. 
The y-axis indicates the periods(frequencies) selected for analysis. 
Bright areas indicate a high Wavelet power around this period(frequency) at that time of the signal. Some synthetic signals
for demonstrational purposes can be found in ``` /example_data/synth_signal.csv ```.

Set a new ```Maximal Power``` and hit ```Update Plot``` to rescale the heatmap if needed.

The *cone of influence* (COI) can be plotted on top of the spectrum by checking the
respective box. 

![WaveletSpectrum overview][spectrum]

[spectrum]:./doc/spectrum.png?raw=true


#####  Ridge Analysis #####

To extract intantaneous frequency and associated phase, a 1d-*ridge* (a profile) has to be traced through the 
2d-power spectrum:

```math
f = ridge(t)
```
This maps **one** frequency (or period) to **one** time point.

The simplest way is to connect all time-consecutive power-maxima. This is what
``` Detect Maximum Ridge ``` does. This works well for all of the examples found in 
``` /data_examples/synth_signal.csv ```.

To exclude parts of the spectrum whith 
low Wavelet power, indicating that no oscillations wihtin the chosen period(frequency)
range are present at that time, set a ``` Power Threshold ```. The actual ridge is indicated as a
red line in spectrum plot, note that no default ridge is shown in a fresh 
``` Wavelet Spectrum ``` window. For a quick check hit the ``` Detect Maximum Ridge ``` button. 
You can also smooth the ridge if needed.

##### Ridge Results #####



Once it is found, the complex Wavelet transform can be evaluated *along*
that ridge yielding a complex time series: $`z(t)`$. 

```math
z(t) = \mathcal{W}_\Psi[s](t, ridge(t) )
``` 
``` Plot Results ``` will then show the extracted
instantaneous periods(frequencies), the phase and power profile along the ridge:

```math
period(t) = 1/ridge(t)
```
```math
phase(t) = arg[z(t)]
```
```math
power(t) = abs[z(t)]
```
![Readout overview][readout]

[readout]:./doc/readout.png?raw=true
