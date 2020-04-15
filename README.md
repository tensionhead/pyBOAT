## pyBOAT - Time Frequency Analysis UI  ##


Tools for time-frequency analysis with Morlet Wavelets
Inspired by 'A Practical Guide to Wavelet Analysis' (Torrence
and Compo 1998), 'Identification of Chirps with Continuous Wavelet Transform'
(Carmona, Hwang and Torresani 1995)
and [The Scientist and Engineer's Guide to Digital Signal Processing](http://www.dspguide.com/).

Version 0.6 June 2019, Frieda Sorgenfrei and Gregor Mönke. 

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

### Installation and Requirements ###

The program needs some scientific python libraries (detailed list will come), it's most
convenient to install [Anaconda 3] (https://conda.io/docs/user-guide/install/download.html) to
get all required Python libraries.

No real 'installation' yet, just download (or clone) the
repository (it's the tiny button sporting a cloud icon on the right).


##### Mac OS #####

After downloading the repository, double click the 
``` TFApy_MacOS.command ``` file. It will open a 
terminal in the background and runs the TFApy program.
You might have to 'allow' 3rd-party apps to run, this
can be done for **El Capitan** by:

``` System Preferences -> Security & Privacy -> Allow Apps downloaded from -> Anywhere ```

For the newest version **Sierra** do a right click on that file,
and choose open.

##### Anaconda troubleshooting #####

In case of errors from Anaconda, you can try to update
your installation by typing

```conda update --all ```

in the terminal.

##### Linux #####

Just run ```python3 TFApy.py ``` on the terminal 
from the ``` /tfapy ``` directory.

##### Windows #####

Everything is a bit more complicated here, so no 'double-clickable' file yet. 
With the windows command line navigate to the``` /tfapy ``` directory
and run ```python3 TFApy.py ```. For some people double-clicking the ```TFApy.py ```
does the trick.

### Usage ###
-------------

##### Data import #####

Just open your saved time-series data by using ``` Open ``` 
from the (small) main window. Supported input formats are:
``` .xls, .xlsx, .csv, .tsv and .txt ```. For other file
extensions, white space separation of the data is assumed.
Please see examples of the supported formats in the 
``` data_examples ``` directory.

After successful import, you can simply click on the table representing
your data to select a specific time-series in the ``` DataViewer ```. 
Alternatively, select a specific time-series from the drop-down menu in the upper left.
To get the correct numbers/units you can change the sampling interval 
and unit name in the top line of the ``` DataViewer ```.

##### Detrending  #####


The featured sinc-detrending is an optimal high-pass filter and removes low frequencies (high periods) 
from the signal via a sharp ``` cut-off-period ```. Details of the implementation can be found at 
[The Scientist and Engineer's Guide to Digital Signal Processing](http://www.dspguide.com/).
Click ``` Refresh Plot ``` and check the ``` Trend ``` and/or ``` Detrended Signal ``` checkbox(es)
to see the effect of the filter on the selected time series.

#### Set up Wavelet Analysis ####

Set the parameters for the Wavelet Analysis in the lower right:

| Input Field   | Meaning    |
| --- | --- |
| Smallest Period | Lower period bound <br> (Nyquist limit built-in)  |
| Number of Periods | Resolution of the transform <br> or number of convolutions             |   
| Highest Period | Upper period bound <br> Should not exceed observation time     |
| Expected maximal power | Upper bound for the colormap <br> indicating the Wavelet power spectrum <br> normalized to white noise |

Leave the ``` Use the detrended signal ``` box checked if you want to use the sinc-detrending. 
``` Analyze Signal ``` will perform the Wavelet transforms of as signal *s* given the Morlet-Wavelet function $`\Psi`$:

```math
\mathcal{W}_\Psi[s](t,f) = \Psi(t,f) * s(t)
```

Here *f* denotes the frequencies of the Wavelets used, hence the implemented Wavelet transform can be viewed as stacked
convolutions of the signal with (complex) Wavelets of the different selected frequencies (periods). When the transform is done, 
a ```Wavelet Spectrum``` window will open.


#### Wavelet Power Spectrum  ####

The input signal for the Wavelet analysis and the resulting power spectrum are shown with aligned time axis. 
The y-axis indicates the periods(frequencies) selected for analysis. The absolute value of the complex 
Wavelet transform, often called the 'power', is given by:

```math
power(t,f) = \frac{| \mathcal{W}_\Psi[s](t,f) |^2}{\sigma^2} 
```

This defines a 2d-spectrum in frequency (period) and time. Dividing by the standard deviation
normalizes the Wavelet power with respect to white noise, which hence has the expected power of one.
Bright areas indicate a high Wavelet power around this period(frequency) at that time of the signal. Some synthetic signals
for demonstrational purposes can be found in ``` /data_examples/synth_signal.csv ```.

####  Ridge Analysis ####

To extract intantaneous frequency and associated phase, a 1d-*ridge* (a profile) has to be traced through the 
2d-power spectrum:

```math
f = ridge(t)
```
This maps **one** frequency (or period) to **one** time point.

##### Maximum Ridge #####

The simplest way is to connect all time-consecutive power-maxima. This is what
``` Detect maximum ridge ``` does. This works well for all of the examples found in 
``` /data_examples/synth_signal.csv ```.

##### Rige from Annealing #####

To constrain the ridge for smoothness, a simple *simulated annealing* scheme is implemented. 
The 'initial guess' comprises a straight line corresponding to a signal with constant
instaneous period (frequency) of ``` Initial period guess ```. 'Wiggling' of the optimized
ridge can be controlled by ``` Curvature cost ```. The default value of zero corresponds
to an unconstrained ridge curvature, whereas high values will enforce more straight ridges.
Details can be found in 'Identification of Chirps with Continuous Wavelet Transform'
(Carmona, Hwang and Torresani 1995).

##### Ridge Results #####

To exclude parts of the spectrum whith 
low Wavelet power, indicating that no oscillations wihtin the selected period(frequency)
range are present at that time, set a ``` Power threshold ```. The actual ridge is indicated as a
red line in spectrum plot, note that no default ridge is shown in a fresh 
``` Wavelet Spectrum ``` window. For a quick check hit the ``` Detect maximum ridge ``` button. 
You can also smooth the ridge if needed.

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

