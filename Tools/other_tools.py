from time import gmtime, strftime
from math import ceil
import numpy as np

def print_time(time_in_seconds: float):
    """
    """

    time_printed = strftime("%Hh%Mm%Ss", gmtime(time_in_seconds)) + str(ceil(100*(time_in_seconds%1)))

    if time_printed.startswith('00'):
        time_printed = time_printed[3:]
        if time_printed.startswith('00'):
            time_printed = time_printed[3:]

    return time_printed


def get_phase_offset(phase_name: str, 
                     phase_inc: float, 
                     pulse: int 
                     ) -> float:
    match phase_name:
        case "zero":
            phase_offset = 0
        case "alt":
            # cycle phase with 180° increment: 0° for even pulse, 180° for odd pulse
            phase_offset = 180.0 * (pulse % 2)
        case "cycle":
            phase_offset = phase_inc * pulse
        case "quad":
            # quadratic pulse
            phase_offset = 0.5 * pulse**2 * phase_inc
        case "quadalt":
            # quadratic phase +
            phase_offset = 0.5 * pulse**2 * phase_inc + 180.0 * (pulse % 2)
        case _:
            raise "Unknown phase name '{}'. Currently supported phases are 'zero', 'alt', 'cycle', 'quad' and 'quadalt'.".format(
                phase_name)

    # return phase offset converted to rad, in the range [0, 2pi[
    return (phase_offset / 180 * np.pi) % (2*np.pi)

