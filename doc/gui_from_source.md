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
