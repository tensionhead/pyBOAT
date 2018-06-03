## TFApy - Time Frequency Analysis UI - early Beta :rocket: ##


### Features ###

* Optimal sinc filter
* Fourier analysis
* Wavelet analysis 
* Ridge detection
* Phase extraction 

### Things not ready yet ###

* Dedicated results window after ridge detection 
* Results export (you can save the Wavelet Spectrum)
* Synthetic signal generator

### Installation ###

No real 'installation' yet, just download (or clone) the
repository and run ``` TFApy.py ``` on the terminal 
from the ``` /src ``` directory.

#### Mac OS ####

After downloading the repository, double click the 
``` TFApy_MacOS ``` command file. It will open a 
terminal in the background and runs the TFApy program.

### Usage ###

Just open your saved time series data by using ``` Open ``` 
from the (small) main window. Supported input formats are:
``` .xls, .xlsx, .csv, .tsv and .txt ```. For other file
extensions, white space separation of the data is assumed.
Please see examples of the supported formats in the 
``` data_examples ``` directory.