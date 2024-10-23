#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 02:46:20 2024

@author: arthur
"""


import numpy as np
import matplotlib.pyplot as plt
import time; start=time.time()
from os import chdir
chdir('/home/arthur/Documents/BH/python/Euclidean Schwarzschild and Reissner-Nordstrom/ERN/GitHub/pypi')

from functions import orbit_BR, orbit, shadow, deflection, shadow4gif, make_gif, DatFile4gif, make_gif_with_DatFile



#Orbits:
Mass=1; Charge=1; Tau=50; N=10000; lim=10;
M=1; Q=Charge/Mass; IniConds=[4,np.pi/2,0,0,0,0.19];
for Type in ["Lorentzian","Euclidean"]:
    Vecc=orbit(Type,Mass,Charge,Tau,N,IniConds,lim)
    
    R=Vecc[:,0]; theta=Vecc[:,1]; phi=Vecc[:,2];
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100); M2=Mass*(1+np.sqrt(1-Q**2));
    x = M2 * np.outer(np.cos(u), np.sin(v))
    y = M2 * np.outer(np.sin(u), np.sin(v))
    z = M2 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z)
    ax.plot(R*np.sin(theta)*np.cos(phi),R*np.sin(theta)*np.sin(phi),R*np.cos(theta));
    
    ax.set_aspect('equal')
    plt.title("Orbit in signature "+Type+" with M="+str(Mass)+", Q="+str(Q))
    plt.show()
    

Charge=1/2; mass=1; charge=1/2; Tau=20; N=10000; lim=20;
IniConds=[7.5,np.pi/2,0,0,0,2];
for epsilon in [-1,1]:
    Vecc=orbit_BR(epsilon,Charge,mass,charge,Tau,N,IniConds,lim)
    R=Vecc[:,0]; theta=Vecc[:,1]; phi=Vecc[:,2];

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = (mass+np.sqrt(mass**2-charge**2)) * np.outer(np.cos(u), np.sin(v))
    y = (mass+np.sqrt(mass**2-charge**2)) * np.outer(np.sin(u), np.sin(v))
    z = (mass+np.sqrt(mass**2-charge**2)) * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z)
    ax.plot(R*np.sin(theta)*np.cos(phi),R*np.sin(theta)*np.sin(phi),R*np.cos(theta));

    ax.set_aspect('equal')
    plt.title("Bertotti-Robinson orbit with eps="+str(epsilon)+", Q="+str(Charge)+", m="+str(mass)+", q="+str(charge))
    plt.show()



#Shadows:
Mass=1/2; Charge=1/2; v=1.5; Image='figure32.png'
shadow("Euclidean",Mass,Charge,v,Image)
shadow("Lorentzian",Mass,Charge,1,Image)
shadow("Lorentzian",Mass,Charge,0.95,Image)


#Deflections
deflection("Euclidean",Mass,Charge,1.5,100)
deflection("Lorentzian",Mass,Charge,1,100)
deflection("Lorentzian",Mass,Charge,0.95,100)




###Comet plots of orbit:
from matplotlib.animation import FuncAnimation

Mass=1; Charge=1; Tau=20; N=150; lim=20;
M=1; Q=Charge/Mass; IniConds=[4,np.pi/2,0,0,0,0.19];

i=0;
for Type in ["Lorentzian","Euclidean"]:
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    hor=Mass*(1+np.sqrt(1-Q**2));
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100);
    x = hor * np.outer(np.cos(u), np.sin(v))
    y = hor * np.outer(np.sin(u), np.sin(v))
    z = hor * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z)
    li,=ax.plot([],[],animated=True,label=Type)
    def init():
        li.set_data([],[])
        return [li]
    Vec=orbit(Type,Mass,Charge,Tau,N,IniConds,lim)
    R=Vec[:,0]; theta=Vec[:,1]; phi=Vec[:,2];
    X0=R*np.sin(theta)*np.cos(phi); Y0=R*np.sin(theta)*np.sin(phi); Z0=R*np.cos(theta);
    ax.set(xlim=[min(X0),max(X0)],ylim=[min(Y0),max(Y0)],zlim=[min(Z0)-1,max(Z0)+1])
    ax.set_aspect('equal')
    def update(frame):
        Xl=X0[:frame+1]; Yl=Y0[:frame+1]; Zl=Z0[:frame+1]
        li.set_data(Xl,Yl)
        li.set_3d_properties(Zl)
        return [li]
    ani=FuncAnimation(fig, update, frames=range(len(X0)), init_func=init, blit=True, interval=2, repeat=False)
    plt.legend()
    ani.save("comet_Reissner-Nordtrom"+Type+".gif", writer='ffmpeg',fps=30)#writer=imagemagick
    plt.show()
    i+=1



Charge=1/2; mass=1; charge=1/2; Tau=12; N=100; lim=20;
IniConds=[7.5,np.pi/2,0,0,0,2];

i=0
for epsilon in [-1,1]:
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    hor=(mass+np.sqrt(mass**2-charge**2));
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100);
    x = hor * np.outer(np.cos(u), np.sin(v))
    y = hor * np.outer(np.sin(u), np.sin(v))
    z = hor * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z)
    li,=ax.plot([],[],animated=True,label="epsilon="+str(epsilon))
    def init():
        li.set_data([],[])
        return [li]
    Vec=orbit_BR(epsilon,Charge,mass,charge,Tau,N,IniConds,lim)
    R=Vec[:,0]; theta=Vec[:,1]; phi=Vec[:,2];
    X0=R*np.sin(theta)*np.cos(phi); Y0=R*np.sin(theta)*np.sin(phi); Z0=R*np.cos(theta);
    ax.set(xlim=[min(X0),max(X0)],ylim=[min(Y0),max(Y0)],zlim=[min(Z0)-1,max(Z0)+1])
    ax.set_aspect('equal')
    def update(frame):
        Xl=X0[:frame]; Yl=Y0[:frame]; Zl=Z0[:frame]
        li.set_data(Xl,Yl)
        li.set_3d_properties(Zl)
        return [li]
    ani=FuncAnimation(fig, update, frames=range(len(X0)), init_func=init, blit=True, interval=2, repeat=False)
    plt.legend()
    ani.save("comet_Bertotti-Robinson"+str(epsilon)+".gif", writer='ffmpeg',fps=30)
    plt.show()
    i+=1




#Making gifs:
Mass=1/2; Charge=1/2; v=1;
Nimages=240; Image="figure.png"; Resol=[60,60]; Shifts=[0,0,3.5]; Direction="d2-"; FPS=24;
for Type in ["Lorentzian","Euclidean"]:
    Name="figure"+Type;
    #make_gif(Nimages,Name,Image,Resol,Shifts,Direction,FPS,Type,Mass,Charge,v)
    DatFile4gif(Resol,Type,Mass,Charge,v)
    make_gif_with_DatFile(Nimages,Name,Image,Resol,Shifts,Direction,FPS,Type,Mass,Charge,v)


print(time.time()-start)