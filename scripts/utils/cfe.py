"""
CFE related functions, e.g. theoretical sensitivity function
"""

import model_config as mcf
import numpy as np

def getCO2effect_simple(airT, co2, co2_ref=None):
    """
    Simple CO2 fertilization effects based on a constant ci/ca value
    K-model 

    From Keenan et al., 2021

    Args:
        airT: air temperature in C
        co2: co2 concentration in ppm
        co2_ref: reference co2 concentration in ppm
    
    Output:
        co2_scalar: co2 effect relative to reference CO2 value (co2_ref)
    """

    # constant
    R = 8.314  # molar gas constant J mol-1 K-1
    r25 = 42.75 # photorespiration point at 25C, converted to ppm
    deltaH = 37830 # activation energy for gamma star J mol-1
    xi = 0.7 # cica = 0.7

    if co2_ref is None:
        co2_ref = mcf.co2_2001 # reference CO2 concentration in 2001: average of MLO and SPO: 370.1; MLO: 371.32
    
    #  calculate gamma star
    expF = (airT - 25) / (R * (airT + 273.15) * 298.15)
    gstar = r25 * np.exp(deltaH * expF)

    # calculate the CO2 scalar
    # this scales GPP to present CO2 to a reference CO2 (2001 value)
    ciRef = xi * co2_ref
    CO2scalarRef = (ciRef - gstar) / (ciRef + 2 * gstar)
    ci = xi * co2
    CO2scalar = (ci - gstar) / (ci + 2 * gstar)

    CO2effect = 1 + (CO2scalar - CO2scalarRef) / CO2scalarRef

    return CO2effect

