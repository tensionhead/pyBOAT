## TFApy - Time Frequency Analysis UI - early Beta :rocket: ##
===============================================================

Tools for time-frequency analysis with Morlet Wavelets
Inspired by 'A Practical Guide to Wavelet Analysis' (Torrence
and Compo 1998), 'Identification of Chirps with Continuous Wavelet Transform'
(Carmona, Hwang and Torresani 1995)
and [The Scientist and Engineer's Guide to Digital Signal Processing](http://www.dspguide.com/).

Version 0.4 June 2018, Gregor Mönke and Frieda Sorgenfrei. 

Questions etc. please to gregor.moenke@embl.de.

**Contributers are welcome!**

### Features ###
-----------------
* Optimal sinc filter
* Fourier analysis
* Wavelet analysis 
* Ridge detection
* Phase extraction 

### Things not ready yet ###
----------------------------
* Dedicated results window after ridge detection 
* Results export (you can save the Wavelet Spectrum)
* Synthetic signal generator

### Installation and Requirements ###
-------------------------------------

The program needs some scientific python libraries (detailed list will come), it's most
convenient to install [Anaconda 3] (https://conda.io/docs/user-guide/install/download.html) to
get all required Python libraries.

No real 'installation' yet, just download (or clone) the
repository and run ``` TFApy.py ``` on the terminal 
from the ``` /src ``` directory.

##### Mac OS #####

After downloading the repository, double click the 
``` TFApy_MacOS ``` command file. It will open a 
terminal in the background and runs the TFApy program.

### Usage ###
-------------

##### Data import #####

Just open your saved time-series data by using ``` Open ``` 
from the (small) main window. Supported input formats are:
``` .xls, .xlsx, .csv, .tsv and .txt ```. For other file
extensions, white space separation of the data is assumed.
Please see examples of the supported formats in the 
``` data_examples ``` directory.

##### Detrending and Analysis Setup #####

After successful import, you can simply click on the table representing
your data to select a specific time-series in the ``` DataViewer ```. 
Alternatively, select a specific time-series from the drop-down menu in the upper left.
To get the correct numbers/units you can change the sampling interval 
and unit name in the top line of the ``` DataViewer ```.
To apply the sinc-filter detrending, enter a (positive) Number for the ``` cut-off-period ```. 
All periods bigger than the one entered will be removed from the data. Click ``` Refresh Plot ```
and check the ``` Trend ``` and/or ``` Detrended Signal ``` checkbox(es) to see the effect of the filter.

Finally, set the parameters for the Wavelet Analysis in the lower right:

| Input Field | Meaning  |
| :------------ |:---------------:| -----:|
| Smallest Period | Lower period bound <br> (Nyquist limit built-in)  |
| Number of Periods | Resolution of the transform <br> or number of convolutions             |   
| Highest Period | Upper period bound <br> Should not exceed observation time     |
| Expected maximal power | Upper bound for the colormap <br> of the Wavelet power spectrum <br> normalized to white noise |
