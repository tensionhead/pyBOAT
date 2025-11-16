"""Some UI defaults"""

# --- the analysis parameter dictionary with defaults ---

default_par_dict = {
    "dt": 1,
    "time_unit": "min",
    "cut_off_period": None,
    "window_size": None,
    "Tmin": None,
    "Tmax": None,
    "nT": 200,
    "float_format": "%.3f",
    "graphics_format": "png",
    "data_format": "csv",
}


# -- reanalyze throttling in ms --

debounce_ms = 100
