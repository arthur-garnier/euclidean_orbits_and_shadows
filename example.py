#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 18:57:29 2024

@author: arthur
"""

import numpy as np
from os import chdir
chdir('/home/arthur/Documents/BH/python/Euclidean Schwarzschild and Reissner-Nordstrom/ERN/GitHub')

from functions import orbit_ERN,orbit_BR,shadow_ERN,shadow_LRN,deflection_ERN,deflection_LRN

#Orbits:
Mass=1; Charge=1; Tau=50; N=10000; lim=10;
M=1; Q=Charge/Mass; IniConds=[1.1,np.pi/2,0,0,0,0.15]; #IniConds=[1.1,0,0,1];
orbit_ERN(Mass,Charge,Tau,N,IniConds,lim)

Charge=1/2; mass=1; charge=1/2;
IniConds=[0.75,np.pi/2,0,0,0,1];
for epsilon in [-1,1]:
    orbit_BR(epsilon,Charge,mass,charge,Tau,N,IniConds,lim)


#Shadows:
Mass=1; Charge=1; v=1.5; Image='figure100.png'
shadow_ERN(Mass,Charge,v,Image)
shadow_LRN(Mass,Charge,1,Image)
shadow_LRN(Mass,Charge,0.95,Image)


#Deflections
deflection_ERN(Mass,Charge,1.5,100)
deflection_LRN(Mass,Charge,1,100)
deflection_LRN(Mass,Charge,0.95,100)