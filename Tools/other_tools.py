from time import gmtime, strftime
from math import ceil
import numpy as np

def print_time(time_in_seconds: float):
    """
    Converts a time in seconds to a human-readable string format "HhMmSsCC",
    where CC is the hundredths of a second.

    Parameters
    ----------
    time_in_seconds : float
        Time duration in seconds.

    Returns
    -------
    time_printed : str
        Formatted time string. Leading "00h" or "00h00m" is removed if unnecessary.
    """
    time_printed = strftime("%Hh%Mm%Ss", gmtime(time_in_seconds)) + str(ceil(100*(time_in_seconds%1)))

    if time_printed.startswith('00'):
        time_printed = time_printed[3:]
        if time_printed.startswith('00'):
            time_printed = time_printed[3:]

    return time_printed

