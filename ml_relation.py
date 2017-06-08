#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 09:36:12 2017

@author: birnstiel
"""

import numpy as np
from scipy.optimize import fsolve
from astropy import constants as c
from astropy import units     as u

def ml_relation(M):
    """
    Returns the approximate Luminosity for a given stellar mass
    
    Arguments:
    ----------
    M : float
        stellar mass in solar masses
    
    Returns:
    --------
    L : float
        stellar luminosity in solar luminosities
    """
    if M<0.43:
        L=0.23*M**2.3
    elif M<2:
        L=M**4
    elif M<20:
        L=1.5*M**3.5
    elif M>20:
        L=3200*M
    else:
        raise ValueError('Invalid stellar mass')
    return L

def lm_relation(L):
    """
    Returns the approximate mass for a given stellar luminosity
    
    Arguments:
    ----------
    L : float
        stellar luminosity in solar luminosities

    
    Returns:
    --------
    
    M : float
        stellar mass in solar masses

    """

    if np.isnan(L):
        return np.nan
    else:
        res=fsolve(lambda m: ml_relation(m)-L,0.5)
        if len(res)==1:
            return res[0]
        else:
            return np.nan
        
def get_semimajoraxis(srad,steff,pds):
    """
    for a list of stellar radii (in solar radii) and effective temperatures,
    estimate the stellar masses and calculate the planetary semimajoraxes in AU
    given the planets periods in days.
    
    Arguments:
    ----------
    
    srad : float array
        stellar radii in solar radii
    
    steff : float array
        stellar effective temperature in K
        
    pds : float array
        planetary period in days
        
    Returns:
    --------
    
    semi : float array
        semimajor axes in AU
        
    Example:
    --------
    >>>get_masses.get_semimajoraxis(1,5778.0,365.25)
    0.99998927361887924
    """
    
    slum  = c.sigma_sb.cgs.value * 4*np.pi*(srad*c.R_sun.cgs.value)**2 * steff**4
    L_sun = c.L_sun.cgs.value
    M_sun = c.M_sun.cgs.value
    G     = c.G.cgs.value
    day   = u.day.in_units('s')
    AU    = c.au.cgs.value
    
    
    smass = np.array([lm_relation(_slum/L_sun) for _slum in np.array(slum,ndmin=1)])
    semi  = (G*smass*M_sun*((pds*day)/(2*np.pi))**2)**(1./3.)/AU
    
    if len(semi)==1:
        return semi[0]
    else:
        return semi