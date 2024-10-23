#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 02:42:41 2024

@author: arthur
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import cmath
from scipy.optimize import fsolve
from scipy.optimize import newton as nt
import matplotlib as mpl

#from scipy.misc import derivative
import cv2 as cv

#from matplotlib.colors import ListedColormap

import os
path0='.';

import pickle
import imageio


def orbit_BR(epsilon,Charge,mass,charge,Tau,N,IniConds,lim):
    Q=Charge; m=mass; q=charge; eps=epsilon; IC=IniConds; tau=Tau; dtau=tau/N;
    
    def From_spherical(R,Rp,T,Tp,P,Pp):
        x=R*np.sin(T)*np.cos(P);
        y=R*np.sin(T)*np.sin(P);
        z=R*np.cos(T);
        xp=(Tp*np.cos(T)*np.cos(P)*R-np.sin(T)*(Pp*np.sin(P)*R-Rp*np.cos(P)));
        yp=(Tp*np.cos(T)*np.sin(P)*R+np.sin(T)*(Pp*np.cos(P)*R+Rp*np.sin(P)));
        zp=Rp*np.cos(T)-R*Tp*np.sin(T);
        BB=[x,xp,y,yp,z,zp];
        return(BB)
    
    def To_spherical(x,xp,y,yp,z,zp):
        P=np.arctan2(y,x);
        R=np.sqrt(x**2+y**2+z**2);
        T=np.arccos(z/R);
        Rp=(x*xp+y*yp+z*zp)/R;
        Tp=(z*Rp-zp*R)/(R*np.sqrt(R**2-z**2));
        Pp=(yp*x-xp*y)/(x**2+y**2);
        XX=[R,Rp,T,Tp,P,Pp];
        return(XX)
    
    def rot(axe,theta,u):
        KK=np.array([[0,-axe[2],axe[1]],[axe[2],0,-axe[0]],[-axe[1],axe[0],0]]); KK=KK/lin.norm(axe,2);
        RR=np.identity(3)+np.sin(theta)*KK+(1-np.cos(theta))*(KK.dot(KK));
        v=RR.dot(u);
        return(v)
    
    X=IC; r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5];
    BB=From_spherical(r,rp,th,thp,ph,php); BBi=np.array([BB[0],BB[2],BB[4]]); BBv=np.array([BB[1],BB[3],BB[5]]);
    x=BBi[0]; y=BBi[1]; z=BBi[2]; vx=BBv[0]; vy=BBv[1]; vz=BBv[2];
    if (abs(z)<1e-10 and abs(vz)<1e-10):
       th0=0; O0=np.array([1,0,0]);
    elif (abs(z)<1e-10 and abs(vz)>=1e-10):
       P0=np.array([y*(vx*y-vy*x)/(x**2+y**2),x*(-vx*y+vy*x)/(x**2+y**2),vz]); Q0=np.array([y*(vx*y-vy*x)/(x**2+y**2),x*(-vx*y+vy*x)/(x**2+y**2),0]);
       th0=np.sign(vz)*np.arccos((np.inner(P0,Q0))/(lin.norm(P0)*lin.norm(Q0))); O0=BBi;
    elif (abs(z)>=1e-10 and abs(vz)<1e-10):
       th0=np.pi/2; O0=BBv;
    else:
       O0=np.array([x-z*vx/vz,y-z*vy/vz,0]); P0=np.array([vy*z/vz-y,-vx*z/vz+x,0]);
       Q0=np.array([-z*(-vy*z+vz*y)*(-vx*y+vy*x)/((vx**2+vy**2)*z**2-2*vz*(vx*x+vy*y)*z+vz**2*(x**2+y**2)),z*(-vx*z+vz*x)*(-vx*y+vy*x)/((vx**2+vy**2)*z**2-2*vz*(vx*x+vy*y)*z+vz**2*(x**2+y**2)),z]);
       th0=np.sign(z)*np.arccos((np.inner(P0,Q0))/(lin.norm(P0)*lin.norm(Q0)));
    
    BBi=rot(O0,-th0,BBi); BBv=rot(O0,-th0,BBv);
    CC=To_spherical(BBi[0],BBv[0],BBi[1],BBv[1],BBi[2],BBv[2]);
    R=CC[0]; Rp=CC[1]; P=CC[4]; Pp=CC[5];# T=CC[2]; Tp=CC[3];
    D=1/R**2-2*m/R+q**2;
    H=Q**2*(eps*D+D**(-1)*Rp**2/R**4+Pp**2); #E=D/np.sqrt(H);
    L=Pp/np.sqrt(H); l=L**2-1/Q**2; Ta=np.arange(0,tau,dtau);
    if l>0:
        Vec=np.array([1/(m+(1/R-m)*np.cos(Ta*np.sqrt(l*H))-Rp/R**2*np.sin(Ta*np.sqrt(l*H))/np.sqrt(l*H)),P+Pp*Ta*np.sqrt(H)]);
    else:
        Vec=np.array([1/(m+(1/R-m)*np.cosh(Ta*np.sqrt(-l*H))-Rp/R**2*np.sinh(Ta*np.sqrt(-l*H))/np.sqrt(-l*H)),P+Pp*Ta*np.sqrt(H)]);
    si=len(Ta);
    Vecc=np.zeros((0,3)); test=0; ii=0;
    while (test==0 and ii<si):
        Rf=Vec[0,ii]; Pf=Vec[1,ii];
        if (Rf<0 or Rf>lim*(m+np.sqrt(m**2-q**2))):
            test=1;
        Cf=rot(O0,th0,np.array([Rf*np.cos(Pf),Rf*np.sin(Pf),0]));
        Vecc=np.vstack([Vecc,np.array([Rf,np.arccos(Cf[2]/Rf),np.arctan2(Cf[1],Cf[0])])]); ii+=1;
    
    return Vecc






def orbit(Type,Mass,Charge,Tau,N,IniConds,lim):
    Rs=2*Mass; IC=[1/Mass*IniConds[0],IniConds[1],IniConds[2],IniConds[3],IniConds[4]*Mass,IniConds[5]*Mass];
    tau=Tau/Mass; dtau=tau/N; Q=Charge/Mass;
    def carlson(x,y,z):
        rtol=1e-10; xn=x; yn=y; zn=z; A0=(xn+yn+zn)/3; m=0;
        Q=np.power(3*rtol,-1/6)*max(abs(A0-xn),abs(A0-yn),abs(A0-zn)); A=A0;
        while Q/(4**m)>abs(A):
            sqx=cmath.sqrt(xn); sqy=cmath.sqrt(yn); sqz=cmath.sqrt(zn);
            if np.real(sqx)<0:
                sqx=-sqx;
            if np.real(sqy)<0:
                sqy=-sqy;
            if np.real(sqz)<0:
                sqz=-sqz;
            lm=sqx*sqy+sqx*sqz+sqy*sqz; A=(A+lm)/4;
            xn=(xn+lm)/4; yn=(yn+lm)/4; zn=(zn+lm)/4; m=m+1;
        X=(A0-x)/(4**m*A); Y=(A0-y)/(4**m*A); Z=-X-Y;
        E2=X*Y-Z**2; E3=X*Y*Z;
        app=(1-E2/10+E3/14+E2**2/24-3*E2*E3/44)/cmath.sqrt(A);
        return(app)
    
    def From_spherical(R,Rp,T,Tp,P,Pp):
        x=R*np.sin(T)*np.cos(P);
        y=R*np.sin(T)*np.sin(P);
        z=R*np.cos(T);
        xp=(Tp*np.cos(T)*np.cos(P)*R-np.sin(T)*(Pp*np.sin(P)*R-Rp*np.cos(P)));
        yp=(Tp*np.cos(T)*np.sin(P)*R+np.sin(T)*(Pp*np.cos(P)*R+Rp*np.sin(P)));
        zp=Rp*np.cos(T)-R*Tp*np.sin(T);
        BB=[x,xp,y,yp,z,zp];
        return(BB)
    
    def To_spherical(x,xp,y,yp,z,zp):
        P=np.arctan2(y,x);
        R=np.sqrt(x**2+y**2+z**2);
        T=np.arccos(z/R);
        Rp=(x*xp+y*yp+z*zp)/R;
        Tp=(z*Rp-zp*R)/(R*np.sqrt(R**2-z**2));
        Pp=(yp*x-xp*y)/(x**2+y**2);
        XX=[R,Rp,T,Tp,P,Pp];
        return(XX)
    
    def rot(axe,theta,u):
        KK=np.array([[0,-axe[2],axe[1]],[axe[2],0,-axe[0]],[-axe[1],axe[0],0]]); KK=KK/lin.norm(axe,2);
        RR=np.identity(3)+np.sin(theta)*KK+(1-np.cos(theta))*(KK.dot(KK));
        v=RR.dot(u);
        return(v)
    
    X=IC; r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5];
    BB=From_spherical(r,rp,th,thp,ph,php); BBi=np.array([BB[0],BB[2],BB[4]]); BBv=np.array([BB[1],BB[3],BB[5]]);
    x=BBi[0]; y=BBi[1]; z=BBi[2]; vx=BBv[0]; vy=BBv[1]; vz=BBv[2];
    if (abs(z)<1e-10 and abs(vz)<1e-10):
       th0=0; O0=np.array([1,0,0]);
    elif (abs(z)<1e-10 and abs(vz)>=1e-10):
       P0=np.array([y*(vx*y-vy*x)/(x**2+y**2),x*(-vx*y+vy*x)/(x**2+y**2),vz]); Q0=np.array([y*(vx*y-vy*x)/(x**2+y**2),x*(-vx*y+vy*x)/(x**2+y**2),0]);
       th0=np.sign(vz)*np.arccos((np.inner(P0,Q0))/(lin.norm(P0)*lin.norm(Q0))); O0=BBi;
    elif (abs(z)>=1e-10 and abs(vz)<1e-10):
       th0=np.pi/2; O0=BBv;
    else:
       O0=np.array([x-z*vx/vz,y-z*vy/vz,0]); P0=np.array([vy*z/vz-y,-vx*z/vz+x,0]);
       Q0=np.array([-z*(-vy*z+vz*y)*(-vx*y+vy*x)/((vx**2+vy**2)*z**2-2*vz*(vx*x+vy*y)*z+vz**2*(x**2+y**2)),z*(-vx*z+vz*x)*(-vx*y+vy*x)/((vx**2+vy**2)*z**2-2*vz*(vx*x+vy*y)*z+vz**2*(x**2+y**2)),z]);
       th0=np.sign(z)*np.arccos((np.inner(P0,Q0))/(lin.norm(P0)*lin.norm(Q0)));
    
    BBi=rot(O0,-th0,BBi); BBv=rot(O0,-th0,BBv);
    CC=To_spherical(BBi[0],BBv[0],BBi[1],BBv[1],BBi[2],BBv[2]);
    R=CC[0]; Rp=CC[1]; P=CC[4]; Pp=CC[5];
    if Type=="Euclidean":
        J=R**2*Pp; H=R**2*Pp**2+R**2*Rp**2/(R**2-2*R+Q**2)+1-2/R+Q**2/R**2; E=(1-2/R+Q**2/R**2)/np.sqrt(H); L=J/np.sqrt(H);
        rpol=np.roots([(1-E**2)/L**2,-2/L**2,Q**2/L**2-1,2,-Q**2]);
        mi=min(abs(rpol-np.real(rpol))); frpol=[np.real(rpol[rr]) for rr in np.where(abs(rpol-np.real(rpol))==mi)][0];
        rbar=frpol[np.where(abs(frpol)==min(abs(frpol)))[0][0]];
        delta=(1-E**2)/L**2; gamma=2*(2*rbar*delta-1/L**2); beta=6*delta*rbar**2-6*rbar/L**2+Q**2/L**2-1; alpha=2*(2*delta*rbar**3-3*rbar**2/L**2+(Q**2/L**2-1)*rbar+1);
    else:
        H=R**2*Pp**2+R**2*Rp**2/(R**2-2*R+Q**2)-(1-2/R+Q**2/R**2);
        if H!=0:
            #J=R**2*Pp; E=(1-2/R+Q**2/R**2)/np.sqrt(H); L=J/np.sqrt(H); mu=-1;
            mu=-1; E=np.sqrt(Rp**2-(1-2/R+Q**2/R**2)*mu+Pp**2*(R**2-2*R+Q**2)); L=R**2*Pp;
        else:
            E=1;
            L=R**2*Pp/(1-2/R+Q**2/R**2); mu=0
        rpol=np.roots([(E**2+mu)/L**2,-2*mu/L**2,-1+mu*Q**2/L**2,2,-Q**2]);
        mi=min(abs(rpol-np.real(rpol))); frpol=[np.real(rpol[rr]) for rr in np.where(abs(rpol-np.real(rpol))==mi)][0];
        rbar=frpol[np.where(abs(frpol)==min(abs(frpol)))[0][0]];
        delta=(E**2+mu)/L**2; gamma=2*(2*delta*rbar-mu/L**2); beta=-1+mu*Q**2/L**2+3*rbar*(gamma-2*delta*rbar); alpha=2+rbar*(2*beta-rbar*(3*gamma-4*delta*rbar));
    g2=(beta**2/3-alpha*gamma)/4; g3=(alpha*beta*gamma/6-alpha**2*delta/2-beta**3/27)/8;
    def weierP(z):
        N0=12;
        zz0=z/(2**N0); zz=1/zz0**2+g2/20*zz0**2+g3/28*zz0**4;
        for j in range(1,N0+1):
            zz=-2*zz+(6*zz**2-g2/2)**2/(4*(4*zz**3-g2*zz-g3));
        return(zz)
    rp2=np.roots([4,0,-g2,-g3]); z0=alpha/(4*(R-rbar))+beta/12;#z0=1/(2*R)-1/12;
    if abs(Rp)<1e-12:
       Z0=carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2])
    else:
       Z0=np.sign(-Rp)*carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2])
    
    def wrs(t):
        return((2-rbar)*(4*np.real(weierP(Z0+t))-beta/3)-alpha);
    
    xxx=fsolve(wrs,0,full_output=False)[0]; vvv=wrs(xxx);
    if (abs(vvv)<1e-8 and abs(np.sign(Pp)*xxx+P)<2*np.pi):
        Z0=-Z0
        tau=min(tau,abs(xxx)); dtau=tau/N;
        
    Vec=4*np.real(weierP(Z0+np.arange(0,tau,dtau)))-beta/3; si=len(Vec);
    Vec=np.array([alpha/Vec+rbar,P+np.sign(Pp)*np.arange(0,tau,dtau)]); Vecc=np.zeros((0,3)); test=0; ii=0;
    while (test==0 and ii<si):
        Rf=Vec[0,ii]; Pf=Vec[1,ii];
        if (Rf<0 or Rf>lim*Rs):
            test=1;
        Cf=rot(O0,th0,np.array([Rf*np.cos(Pf),Rf*np.sin(Pf),0]));
        Vecc=np.vstack([Vecc,np.array([Rs/2*Rf,np.arccos(Cf[2]/Rf),np.arctan2(Cf[1],Cf[0])])]); ii+=1;
    
    return Vecc







def shadow(Type,Mass,Charge,v,Image):
    x0=7; rf=10; Xmax=2.3; Q=Charge/Mass;
    if Type=="Lorentzian":
        if v<1:
            mu=-1
        elif v==1:
            mu=0
        else:
            mu=1
    Img=cv.imread(Image);
    Img=cv.cvtColor(Img,cv.COLOR_BGR2RGB)
    Npix=np.shape(Img)[0]; Npiy=np.shape(Img)[1]; IMG=np.zeros((Npiy,Npix,3));
    for i in range(3):
        IMG[:,:,i]=np.transpose(Img[:,:,i])/256;
    Npix=np.shape(IMG)[0]; Npiy=np.shape(IMG)[1];
    XX=np.linspace(-Xmax,Xmax,Npix); YY=np.linspace(-Xmax*Npiy/Npix,Xmax*Npiy/Npix,Npiy);
    h=x0*Xmax*np.sqrt(1+Npiy**2/Npix**2)/(rf-Xmax*np.sqrt(1+Npiy**2/Npix**2));

    def carlson(x,y,z):
        rtol=1e-10; xn=x; yn=y; zn=z; A0=(xn+yn+zn)/3; m=0;
        Q=np.power(3*rtol,-1/6)*max(abs(A0-xn),abs(A0-yn),abs(A0-zn)); A=A0;
        while Q/(4**m)>abs(A):
            sqx=cmath.sqrt(xn); sqy=cmath.sqrt(yn); sqz=cmath.sqrt(zn);
            if np.real(sqx)<0:
                sqx=-sqx;
            if np.real(sqy)<0:
                sqy=-sqy;
            if np.real(sqz)<0:
                sqz=-sqz;
            lm=sqx*sqy+sqx*sqz+sqy*sqz; A=(A+lm)/4;
            xn=(xn+lm)/4; yn=(yn+lm)/4; zn=(zn+lm)/4; m=m+1;
        X=(A0-x)/(4**m*A); Y=(A0-y)/(4**m*A); Z=-X-Y;
        E2=X*Y-Z**2; E3=X*Y*Z;
        app=(1-E2/10+E3/14+E2**2/24-3*E2*E3/44)/cmath.sqrt(A);
        return(app)
    
    def To_spherical(x,xp,y,yp):
        P=np.arctan2(y,x);
        R=np.sqrt(x**2+y**2);
        Rp=(x*xp+y*yp)/R;
        Pp=(yp*x-xp*y)/(x**2+y**2);
        XX=[R,Rp,P,Pp];
        return(XX)
    
    def rot(axe,theta,u):
        KK=np.array([[0,-axe[2],axe[1]],[axe[2],0,-axe[0]],[-axe[1],axe[0],0]]); KK=KK/lin.norm(axe,2);
        RR=np.identity(3)+np.sin(theta)*KK+(1-np.cos(theta))*(KK.dot(KK));
        v=RR.dot(u);
        return(v)
    
    def projtoplane(w):
        wp=[-1,np.arctan2(w[1],-w[0]),np.pi/2-np.arccos(w[2]/rf)];
        return(wp)
    
    def init_conds(y):
        Z=To_spherical(x0,h/np.sqrt(h**2+y**2),y,-y/np.sqrt(h**2+y**2));
        Z=[Z[0]/Mass,Z[2],Z[1],Z[3]*Mass];
        r=Z[0]; rp=Z[2]; php=Z[3];
        if Type=="Euclidean":
            co=np.sqrt((Q**2*php**2 + php**2*r**2 - 2*php**2*r + rp**2)*((v**2 + 1)*Q**2 + r**2*v**2 + (-2*v**2 - 2)*r))*(Q**2 + r**2 - 2*r)/(((Q**2 + r**2 - 2*r)*php**2 + rp**2)*r**3)
        else:
            if mu!=0:
                co=np.sqrt((Q**2*php**2 + php**2*r**2 - 2*php**2*r + rp**2)*((-v**2 + 1)*Q**2 + 2*r**2-r**2*v**2 + (2*v**2 - 2)*r))*(Q**2 + r**2 - 2*r)/(((Q**2 + r**2 - 2*r)*php**2 + rp**2)*r**3)
            else:
                co=r*np.sqrt((Q**2*php**2 + php**2*r**2 - 2*php**2*r + rp**2))*(Q**2 + r**2 - 2*r)/(((Q**2 + r**2 - 2*r)*php**2 + rp**2)*r**3)
        Z[2]=co*Z[2]; Z[3]=co*Z[3];
        return(Z)
    
    def weierP(g2,g3,z):
        N0=12;
        zz0=z/(2**N0); zz=1/zz0**2+g2/20*zz0**2+g3/28*zz0**4;
        for j in range(N0):
            zz=-2*zz+(6*zz**2-g2/2)**2/(4*(4*zz**3-g2*zz-g3));
        return(zz)
    
    def newton(g2,g3,Z,t):
        def toanihil(s):
            return((rf/Mass-rbar)*(4*np.real(weierP(g2,g3,Z+s))-beta/3)-alpha);
        sgn=np.sign(toanihil(t)); sol=t; step=-0.05;
        while sgn*toanihil(sol)>0:
            sol=sol+step;
        sol=nt(toanihil,sol);
        return(sol)
    
    Xred=np.zeros((0,2));
    AR=[];
    for xx in XX[int(np.floor(Npix/2)):Npix]:
        for yy in YY[int(np.floor(Npiy/2)):Npiy]:
            AR.append(np.sqrt(xx**2+yy**2));
    AR=np.sort(AR)
    for zz in [zu for zu in AR if zu!=0]:
        X=init_conds(zz)
        r=X[0]; ph=X[1]; rp=X[2]; php=X[3];
        if Type=="Euclidean":
            J=r**2*php; H=r**2*php**2+r**2*rp**2/(r**2-2*r+Q**2)+1-2/r+Q**2/r**2; E=(1-2/r+Q**2/r**2)/np.sqrt(H); L=J/np.sqrt(H);
            rpol=np.roots([(1-E**2)/L**2,-2/L**2,Q**2/L**2-1,2,-Q**2]);
        else:
            if mu!=0:
                J=r**2*php; H=r**2*php**2+r**2*rp**2/(r**2-2*r+Q**2)-(1-2/r+Q**2/r**2); E=(1-2/r+Q**2/r**2)/np.sqrt(H); L=J/np.sqrt(H);
                mu=-1; E=np.sqrt(rp**2-(1-2/r+Q**2/r**2)*mu+php**2*(r**2-2*r+Q**2)); L=r**2*php;
            else:
                E=1;
                L=r**2*php;#/(1-2/r+Q**2/r**2)
            rpol=np.roots([(E**2+mu)/L**2,-2*mu/L**2,-1+mu*Q**2/L**2,2,-Q**2]);
        mi=min(abs(rpol-np.real(rpol))); frpol=[np.real(rpol[rr]) for rr in np.where(abs(rpol-np.real(rpol))==mi)][0];
        rbar=frpol[np.where(abs(frpol)==min(abs(frpol)))[0][0]];
        if Type=="Lorentzian":
            delta=(E**2+mu)/L**2; gamma=2*(2*delta*rbar-mu/L**2); beta=-1+mu*Q**2/L**2+3*rbar*(gamma-2*delta*rbar); alpha=2+rbar*(2*beta-rbar*(3*gamma-4*delta*rbar));
        else:
            delta=(1-E**2)/L**2; gamma=2*(2*rbar*delta-1/L**2); beta=6*delta*rbar**2-6*rbar/L**2+Q**2/L**2-1; alpha=2*(2*delta*rbar**3-3*rbar**2/L**2+(Q**2/L**2-1)*rbar+1);
        g2=(beta**2/3-alpha*gamma)/4; g3=(alpha*beta*gamma/6-alpha**2*delta/2-beta**3/27)/8;
        rp2=np.roots([4,0,-g2,-g3]); z0=alpha/(4*(r-rbar))+beta/12;#z0=1/(2*r)-1/12;
        if abs(rp)<1e-12:
           Z0=carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2])
        else:
           Z0=np.sign(-rp)*carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2]);
        new=newton(g2,g3,Z0,0); P=ph+np.sign(php)*new;
        if Type=="Euclidean":
            Xred=np.vstack([Xred,np.array([zz,P])]);
        else:
            if P>np.pi/2:
                Xred=np.vstack([Xred,np.array([zz,P])]);

    Umax=0; Vmax=0; KK=np.zeros((Npix,Npiy));
    for i in range(int(np.floor(Npix/2)),Npix):
        x=XX[i];
        for j in range(int(np.floor(Npiy/2)),Npiy):
            y=YY[j]; r=np.sqrt(x**2+y**2);
            mi=min(abs(r-Xred[:,0]));
            if mi<1e-10:
                k=np.where(abs(r-Xred[:,0])==mi)[0][0];
                KK[i,j]=k; KK[i,Npiy-j-1]=k;
                KK[Npix-i-1,j]=k; KK[Npix-i-1,Npiy-j-1]=k;
    
    Umax=np.pi/2; Vmax=Umax*Npiy/Npix;
    xred=np.zeros((Npix,Npiy,3));
    for i in range(Npix):
        xx=XX[i];
        for j in range(Npiy):
            yy=YY[j];
            if KK[i,j]!=0:
                P=Xred[int(KK[i,j]),1];
                Z=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([np.cos(P),np.sin(P),0]));
                Z=projtoplane(rf*Z);
                t1=(Z[1]+Umax)/(2*Umax); t2=(Z[2]+Vmax)/(2*Vmax);
                s1=abs(1-abs(1-t1)); s2=abs(1-abs(1-t2));
                ii=int(max(1,min(Npix,np.ceil(s1*Npix)))); jj=int(max(1,min(Npiy,np.ceil(s2*Npiy))));
                xred[i,j,0]=IMG[ii-1,jj-1,0]; xred[i,j,1]=IMG[ii-1,jj-1,1]; xred[i,j,2]=IMG[ii-1,jj-1,2];
                
    xredt=np.zeros((Npiy,Npix,3));
    for k in range(3):
        xredt[:,:,k]=np.transpose(xred[:,:,k]);
    
    plt.figure(figsize=(12,12))
    plt.imshow(xredt)
    plt.grid(False)
    plt.axis('off')
    plt.show()







def deflection(Type,Mass,Charge,v,N):
    Q=Charge/Mass;
    if Type=="Lorentzian":
        Mass=0.35*Mass;
        if v<1:
            mu=-1
        elif v==1:
            mu=0
        else:
            mu=1
    r0=100; Xmax=2.3/Mass; Npix=N; Npiy=N; r0=r0/Mass; h=r0/2;
    XX=np.linspace(-Xmax,Xmax,Npix); YY=np.linspace(-Xmax*Npiy/Npix,Xmax*Npiy/Npix,Npiy);
    
    def carlson(x,y,z):
        rtol=1e-10; xn=x; yn=y; zn=z; A0=(xn+yn+zn)/3; m=0;
        Q=np.power(3*rtol,-1/6)*max(abs(A0-xn),abs(A0-yn),abs(A0-zn)); A=A0;
        while Q/(4**m)>abs(A):
            sqx=cmath.sqrt(xn); sqy=cmath.sqrt(yn); sqz=cmath.sqrt(zn);
            if np.real(sqx)<0:
                sqx=-sqx;
            if np.real(sqy)<0:
                sqy=-sqy;
            if np.real(sqz)<0:
                sqz=-sqz;
            lm=sqx*sqy+sqx*sqz+sqy*sqz; A=(A+lm)/4;
            xn=(xn+lm)/4; yn=(yn+lm)/4; zn=(zn+lm)/4; m=m+1;
        X=(A0-x)/(4**m*A); Y=(A0-y)/(4**m*A); Z=-X-Y;
        E2=X*Y-Z**2; E3=X*Y*Z;
        app=(1-E2/10+E3/14+E2**2/24-3*E2*E3/44)/cmath.sqrt(A);
        return(app)

    def init_conds(y):
        th=-np.arctan2(-y,h)
        if Type=="Euclidean":
            E=1/np.sqrt(v**2+1)
            b=r0*np.sin(th)/v*np.sqrt((Q**2*(1+v**2)+r0*v**2*(r0-2)-2*r0)/(Q**2+r0**2-2*r0))
            Z=[r0,b,E,th];
        else:
            if mu!=0:
                E=1/np.sqrt(1-v**2)
                b=r0*np.sin(th)*np.sqrt((Q**2*mu*v**2 + r0**2*mu*v**2 - 2*r0*mu*v**2 - Q**2*mu - r0**2*mu - r0**2 + 2*r0*mu)/((Q**2 + r0**2 - 2*r0)*(mu*v**2-mu-1)))
                L=b*v*E;
            else:
                E=1
                b=r0**2*np.sin(th)/np.sqrt((r0**2+Q**2-2*r0));
                L=b*E
            Z=[r0,E,L,th];
        return Z
    
    Xred=np.zeros((0,2));
    AR=[];
    for xx in XX[int(np.floor(Npix/2)):Npix]:
        for yy in YY[int(np.floor(Npiy/2)):Npiy]:
            AR.append(np.sqrt(xx**2+yy**2));
    AR=np.sort(AR)
    for zz in [zu for zu in AR if zu!=0]:
        if Type=="Euclidean":
            X=init_conds(zz); r=X[0]; b=X[1]; E=X[2]; th=X[3]; L=b*v*E;
            rpol=np.roots([(1-E**2)/L**2,-2/L**2,Q**2/L**2-1,2,-Q**2]);
            mi=min(abs(rpol-np.real(rpol))); frpol=[np.real(rpol[rr]) for rr in np.where(abs(rpol-np.real(rpol))==mi)][0];
            rbar=frpol[np.where(abs(frpol)==min(abs(frpol)))[0][0]];
            delta=(1-E**2)/L**2; gamma=2*(2*rbar*delta-1/L**2); beta=6*delta*rbar**2-6*rbar/L**2+Q**2/L**2-1; alpha=2*(2*delta*rbar**3-3*rbar**2/L**2+(Q**2/L**2-1)*rbar+1);
        else:
            X=init_conds(zz); r=X[0]; E=X[1]; L=X[2]; th=X[3];
            rpol=np.roots([(E**2+mu)/L**2,-2*mu/L**2,-1+mu*Q**2/L**2,2,-Q**2]);
            mi=min(abs(rpol-np.real(rpol))); frpol=[np.real(rpol[rr]) for rr in np.where(abs(rpol-np.real(rpol))==mi)][0];
            rbar=frpol[np.where(abs(frpol)==min(abs(frpol)))[0][0]];
            delta=(E**2+mu)/L**2; gamma=2*(2*delta*rbar-mu/L**2); beta=-1+mu*Q**2/L**2+3*rbar*(gamma-2*delta*rbar); alpha=2+rbar*(2*beta-rbar*(3*gamma-4*delta*rbar));
        g2=(beta**2/3-alpha*gamma)/4; g3=(alpha*beta*gamma/6-alpha**2*delta/2-beta**3/27)/8;
        rp2=np.roots([4,0,-g2,-g3]);
        rmin=max(rpol);
        zmin=alpha/(4*(rmin-rbar))+beta/12; zinf=beta/12; z0=alpha/(4*(r-rbar))+beta/12;
        Rf1=carlson(zinf-rp2[0],zinf-rp2[1],zinf-rp2[2])-carlson(zmin-rp2[0],zmin-rp2[1],zmin-rp2[2]);
        Rf2=carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2])-carlson(zmin-rp2[0],zmin-rp2[1],zmin-rp2[2]);
        if Type=="Euclidean":
            Xred=np.vstack([Xred,np.array([zz,np.real(-(Rf1+Rf2)+th-np.pi)])]);#Xred=np.vstack([Xred,np.array([zz,(-2*(Rf1)-np.pi)])]);
        else:
            if np.real(-(Rf1+Rf2)+th-np.pi)>0:
                Xred=np.vstack([Xred,np.array([zz,np.real(-(Rf1+Rf2)+th-np.pi)])]);

    if Type=="Euclidean":
        cmap=plt.get_cmap('RdBu_r',len(Xred[:,1]))#RdBu_r
    else:
        cmap=plt.get_cmap('Reds',len(Xred[:,1]))
    
    KK=np.zeros((Npix,Npiy));
    for i in range(int(np.floor(Npix/2)),Npix):
        x=XX[i];
        for j in range(int(np.floor(Npiy/2)),Npiy):
            y=YY[j]; r=np.sqrt(x**2+y**2);
            mi=min(abs(r-Xred[:,0]));
            if mi<1e-10:
                k=np.where(abs(r-Xred[:,0])==mi)[0][0];
                KK[i,j]=k; KK[i,Npiy-j-1]=k;
                KK[Npix-i-1,j]=k; KK[Npix-i-1,Npiy-j-1]=k;
    
    xred=np.zeros((Npix,Npiy,3));
    Md=max(Xred[:,1]); md=min(Xred[:,1])
    MD=max(abs(md),abs(Md));

    if Type=="Euclidean":
        norm = mpl.colors.Normalize(vmin=-MD, vmax=MD)
    else:
        norm = mpl.colors.Normalize(vmin=-0*MD, vmax=MD)
    
    for i in range(Npix):
        xx=XX[i];
        for j in range(Npiy):
            yy=YY[j];
            if KK[i,j]!=0:
                if Type=="Euclidean":
                    coe=Xred[int(KK[i,j]),1]; xred0=cmap(int((coe+MD)/(2*MD)*len(Xred[:,1])));
                else:
                    coe=Xred[int(KK[i,j]),1]; xred0=cmap(int((coe+0*MD)/(1*MD)*len(Xred[:,1])));
                xred[i,j,:]=[xred0[0],xred0[1],xred0[2]]

    xredt=np.zeros((Npiy,Npix,3));
    for k in range(3):
        xredt[:,:,k]=np.transpose(xred[:,:,k]);
    
    plt.figure(figsize=(12,12))
    plt.imshow(xredt)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
    sm.set_array([])
    if Type=="Euclidean":
        plt.colorbar(sm, ax=plt.gca(), ticks=np.linspace(-MD,MD, 11+10),label='$\Delta\phi[rad]$',shrink=0.812) 
    else:
        plt.colorbar(sm, ax=plt.gca(), ticks=np.linspace(-0*MD,MD, 11+10),label='$\Delta\phi[rad]$',shrink=0.812)
    plt.grid(False)
    plt.axis('off')
    plt.legend()
    plt.show()
    
    
    





def shadow4gif(Type,Mass,Charge,v,Image_matrix):
    x0=7; rf=10; Xmax=2.3; Q=Charge/Mass;
    if Type=="Lorentzian":
        if v<1:
            mu=-1
        elif v==1:
            mu=0
        else:
            mu=1
    Img=Image_matrix.astype(int)
    Npix=np.shape(Img)[0]; Npiy=np.shape(Img)[1]; IMG=np.zeros((Npiy,Npix,3));
    for i in range(3):
        IMG[:,:,i]=np.transpose(Img[:,:,i]);
    Npix=np.shape(IMG)[0]; Npiy=np.shape(IMG)[1];
    XX=np.linspace(-Xmax,Xmax,Npix); YY=np.linspace(-Xmax*Npiy/Npix,Xmax*Npiy/Npix,Npiy);
    h=x0*Xmax*np.sqrt(1+Npiy**2/Npix**2)/(rf-Xmax*np.sqrt(1+Npiy**2/Npix**2));

    def carlson(x,y,z):
        rtol=1e-10; xn=x; yn=y; zn=z; A0=(xn+yn+zn)/3; m=0;
        Q=np.power(3*rtol,-1/6)*max(abs(A0-xn),abs(A0-yn),abs(A0-zn)); A=A0;
        while Q/(4**m)>abs(A):
            sqx=cmath.sqrt(xn); sqy=cmath.sqrt(yn); sqz=cmath.sqrt(zn);
            if np.real(sqx)<0:
                sqx=-sqx;
            if np.real(sqy)<0:
                sqy=-sqy;
            if np.real(sqz)<0:
                sqz=-sqz;
            lm=sqx*sqy+sqx*sqz+sqy*sqz; A=(A+lm)/4;
            xn=(xn+lm)/4; yn=(yn+lm)/4; zn=(zn+lm)/4; m=m+1;
        X=(A0-x)/(4**m*A); Y=(A0-y)/(4**m*A); Z=-X-Y;
        E2=X*Y-Z**2; E3=X*Y*Z;
        app=(1-E2/10+E3/14+E2**2/24-3*E2*E3/44)/cmath.sqrt(A);
        return(app)
    
    def To_spherical(x,xp,y,yp):
        P=np.arctan2(y,x);
        R=np.sqrt(x**2+y**2);
        Rp=(x*xp+y*yp)/R;
        Pp=(yp*x-xp*y)/(x**2+y**2);
        XX=[R,Rp,P,Pp];
        return(XX)
    
    def rot(axe,theta,u):
        KK=np.array([[0,-axe[2],axe[1]],[axe[2],0,-axe[0]],[-axe[1],axe[0],0]]); KK=KK/lin.norm(axe,2);
        RR=np.identity(3)+np.sin(theta)*KK+(1-np.cos(theta))*(KK.dot(KK));
        v=RR.dot(u);
        return(v)
    
    def projtoplane(w):
        wp=[-1,np.arctan2(w[1],-w[0]),np.pi/2-np.arccos(w[2]/rf)];
        return(wp)
    
    def init_conds(y):
        Z=To_spherical(x0,h/np.sqrt(h**2+y**2),y,-y/np.sqrt(h**2+y**2));
        Z=[Z[0]/Mass,Z[2],Z[1],Z[3]*Mass];
        r=Z[0]; rp=Z[2]; php=Z[3];
        if Type=="Euclidean":
            co=np.sqrt((Q**2*php**2 + php**2*r**2 - 2*php**2*r + rp**2)*((v**2 + 1)*Q**2 + r**2*v**2 + (-2*v**2 - 2)*r))*(Q**2 + r**2 - 2*r)/(((Q**2 + r**2 - 2*r)*php**2 + rp**2)*r**3)
        else:
            if mu!=0:
                co=np.sqrt((Q**2*php**2 + php**2*r**2 - 2*php**2*r + rp**2)*((-v**2 + 1)*Q**2 + 2*r**2-r**2*v**2 + (2*v**2 - 2)*r))*(Q**2 + r**2 - 2*r)/(((Q**2 + r**2 - 2*r)*php**2 + rp**2)*r**3)
            else:
                co=r*np.sqrt((Q**2*php**2 + php**2*r**2 - 2*php**2*r + rp**2))*(Q**2 + r**2 - 2*r)/(((Q**2 + r**2 - 2*r)*php**2 + rp**2)*r**3)
        Z[2]=co*Z[2]; Z[3]=co*Z[3];
        return(Z)
    
    def weierP(g2,g3,z):
        N0=12;
        zz0=z/(2**N0); zz=1/zz0**2+g2/20*zz0**2+g3/28*zz0**4;
        for j in range(N0):
            zz=-2*zz+(6*zz**2-g2/2)**2/(4*(4*zz**3-g2*zz-g3));
        return(zz)
    
    def newton(g2,g3,Z,t):
        def toanihil(s):
            return((rf/Mass-rbar)*(4*np.real(weierP(g2,g3,Z+s))-beta/3)-alpha);
        sgn=np.sign(toanihil(t)); sol=t; step=-0.05;
        while sgn*toanihil(sol)>0:
            sol=sol+step;
        sol=nt(toanihil,sol);
        return(sol)
    
    Xred=np.zeros((0,2));
    AR=[];
    for xx in XX[int(np.floor(Npix/2)):Npix]:
        for yy in YY[int(np.floor(Npiy/2)):Npiy]:
            AR.append(np.sqrt(xx**2+yy**2));
    AR=np.sort(AR)
    for zz in [zu for zu in AR if zu!=0]:
        X=init_conds(zz)
        r=X[0]; ph=X[1]; rp=X[2]; php=X[3];
        if Type=="Euclidean":
            J=r**2*php; H=r**2*php**2+r**2*rp**2/(r**2-2*r+Q**2)+1-2/r+Q**2/r**2; E=(1-2/r+Q**2/r**2)/np.sqrt(H); L=J/np.sqrt(H);
            rpol=np.roots([(1-E**2)/L**2,-2/L**2,Q**2/L**2-1,2,-Q**2]);
        else:
            if mu!=0:
                J=r**2*php; H=r**2*php**2+r**2*rp**2/(r**2-2*r+Q**2)-(1-2/r+Q**2/r**2); E=(1-2/r+Q**2/r**2)/np.sqrt(H); L=J/np.sqrt(H);
            else:
                E=1;
                L=r**2*php/(1-2/r+Q**2/r**2)
            rpol=np.roots([(E**2+mu)/L**2,-2*mu/L**2,-1+mu*Q**2/L**2,2,-Q**2]);
        mi=min(abs(rpol-np.real(rpol))); frpol=[np.real(rpol[rr]) for rr in np.where(abs(rpol-np.real(rpol))==mi)][0];
        rbar=frpol[np.where(abs(frpol)==min(abs(frpol)))[0][0]];
        if Type=="Lorentzian":
            delta=(E**2+mu)/L**2; gamma=2*(2*delta*rbar-mu/L**2); beta=-1+mu*Q**2/L**2+3*rbar*(gamma-2*delta*rbar); alpha=2+rbar*(2*beta-rbar*(3*gamma-4*delta*rbar));
        else:
            delta=(1-E**2)/L**2; gamma=2*(2*rbar*delta-1/L**2); beta=6*delta*rbar**2-6*rbar/L**2+Q**2/L**2-1; alpha=2*(2*delta*rbar**3-3*rbar**2/L**2+(Q**2/L**2-1)*rbar+1);
        g2=(beta**2/3-alpha*gamma)/4; g3=(alpha*beta*gamma/6-alpha**2*delta/2-beta**3/27)/8;
        rp2=np.roots([4,0,-g2,-g3]); z0=alpha/(4*(r-rbar))+beta/12;#z0=1/(2*r)-1/12;
        if abs(rp)<1e-12:
           Z0=carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2])
        else:
           Z0=np.sign(-rp)*carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2]);
        new=newton(g2,g3,Z0,0); P=ph+np.sign(php)*new;
        if Type=="Euclidean":
            Xred=np.vstack([Xred,np.array([zz,P])]);
        else:
            if P>np.pi/2:
                Xred=np.vstack([Xred,np.array([zz,P])]);

    Umax=0; Vmax=0; KK=np.zeros((Npix,Npiy));
    for i in range(int(np.floor(Npix/2)),Npix):
        x=XX[i];
        for j in range(int(np.floor(Npiy/2)),Npiy):
            y=YY[j]; r=np.sqrt(x**2+y**2);
            mi=min(abs(r-Xred[:,0]));
            if mi<1e-10:
                k=np.where(abs(r-Xred[:,0])==mi)[0][0];
                KK[i,j]=k; KK[i,Npiy-j-1]=k;
                KK[Npix-i-1,j]=k; KK[Npix-i-1,Npiy-j-1]=k;
    
    Umax=np.pi/2; Vmax=Umax*Npiy/Npix;
    xred=np.zeros((Npix,Npiy,3));
    for i in range(Npix):
        xx=XX[i];
        for j in range(Npiy):
            yy=YY[j];
            if KK[i,j]!=0:
                P=Xred[int(KK[i,j]),1];
                Z=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([np.cos(P),np.sin(P),0]));
                Z=projtoplane(rf*Z);
                t1=np.real(Z[1]+Umax)/(2*Umax); t2=np.real(Z[2]+Vmax)/(2*Vmax);
                s1=abs(1-abs(1-t1)); s2=abs(1-abs(1-t2));
                ii=int(max(1,min(Npix,np.ceil(s1*Npix)))); jj=int(max(1,min(Npiy,np.ceil(s2*Npiy))));
                xred[i,j,0]=IMG[ii-1,jj-1,0]; xred[i,j,1]=IMG[ii-1,jj-1,1]; xred[i,j,2]=IMG[ii-1,jj-1,2];
                
    xredt=np.zeros((Npiy,Npix,3));
    for k in range(3):
        xredt[:,:,k]=np.transpose(xred[:,:,k]);
    
    return xredt







#------------------------------------------------------------------------------



def DatFile4gif(Resol,Type,Mass,Charge,v):
    N1=Resol[0]; N2=Resol[1];
    Ns=[N1,N2];
    os.chdir(path0);
    img1=np.zeros((N1,N2,3))
    for k in range(3):
        img1[:,:,k]=np.array([k*N1*N2+np.linspace(i*N2,(i+1)*N2-1,N2).astype(int) for i in range(N1)])
    img2=shadow4gif(Type,Mass,Charge,v,img1);
    lis1=np.zeros((0,3)); lis2=np.zeros((0,3))
    for i in range(N1):
        for j in range(N2):
            lis1=np.vstack([lis1,np.array([img1[i,j,0],img1[i,j,1],img1[i,j,2]])])
            lis2=np.vstack([lis2,np.array([img2[i,j,0],img2[i,j,1],img2[i,j,2]])])
    L=[];
    for k in range(N1*N2):
        pix=lis2[k,:]
        l=0;
        while (l<N1*N2 and (lis1[l,0]!=pix[0] or lis1[l,1]!=pix[1] or lis1[l,2]!=pix[2])):
            l+=1
        if l>=N1*N2:
            L.append(-1)
        else:
            L.append(l)
    Ls=list([N1,N2,L])

    path=path0+"/dat_files"
    try:
        os.mkdir(path)
        #print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    os.chdir(path)
    path="./file"+"_"+str(Resol)+"_"+str(Type)+"_"+str(Mass)+"_"+str(Charge)+"_"+str(v)+".dat";

    with open(path, 'wb') as f:
        pickle.dump([Ns,Ls,Type,Mass,Charge,v,img1,img2], f, protocol=-1)
        f.close()
    
    os.chdir('../')



def make_gif_with_DatFile(Nimages,Name,Image,Resol,Shifts,Direction,FPS,Type,Mass,Charge,v):
    N1=Resol[0]; N2=Resol[1]; K1=Shifts[0]; K2=Shifts[1]; coe=Shifts[2]; Ns=[N1,N2];
    img=cv.imread(Image); img=cv.cvtColor(img,cv.COLOR_BGR2RGB); img=img/256;
    N0=int(np.floor((np.shape(img)[0]-N1)/2)); names=[]; name0=Name; ext=".png"
    path=path0+'/'+Name+'_gif';
    try:
        os.mkdir(path)
        #print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    for j in range(Nimages):
        if j<10:
            names.append(name0+"000"+str(j)+ext)
        elif j<100:
            names.append(name0+"00"+str(j)+ext)
        elif j<1000:
            names.append(name0+"0"+str(j)+ext)
        else:
            names.append(name0+str(j)+ext)
    os.chdir(path)
    path=os.getcwd()+"/temp"
    try:
        os.mkdir(path)
        #print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    imgs=list([]);
    os.chdir(path)
    for i0 in range(len(names)):
        img0=np.zeros((N1,N2,3))
        floe1=int(np.floor(coe*i0)+K1); floe2=int(np.floor(coe*i0)+K2);
        for k in range(3):
            if Direction=="d2-":
                img0[:,:,k]=img[floe1:(floe1+N1),floe2:(floe2+N2),k]
            elif Direction=="d2+":
                img0[:,:,k]=img[(floe1+N1):floe1:-1,(floe2+N2):floe2:-1,k]
            elif Direction=="d1-":
                img0[:,:,k]=img[(floe1+N1):floe1:-1,floe2:(floe2+N2),k]
            elif Direction=="d1+":
                img0[:,:,k]=img[floe1:(floe1+N1),(floe2+N2):floe2:-1,k]
            elif Direction=="h-":
                img0[:,:,k]=img[N0+K1:N0+K1+N1,floe2:(floe2+N2),k]
            elif Direction=="h+":
                img0[:,:,k]=img[N0+K1+N1:N0+K1:-1,(floe2+N2):floe2:-1,k]
            elif Direction=="v+":
                img0[:,:,k]=img[floe1:(floe1+N1),N0+K2:N0+K2+N2,k]
            elif Direction=="v-":
                img0[:,:,k]=img[(floe1+N1):floe1:-1,N0+K2+N2:N0+K2:-1,k]
        imgs.append(img0)
        img0*=256
        img0.astype(int)
        cv.imwrite(names[i0],img0)
    
    file="../../dat_files/file"+"_"+str(Resol)+"_"+str(Type)+"_"+str(Mass)+"_"+str(Charge)+"_"+str(v)+".dat";

    try:
        with open(file,'rb') as f:
            [Ns,Ls,Type,Mass,Charge,v,img1,img2] = pickle.load(f)
            f.close()
        
        Imgs0=os.listdir(); Imgs0.sort(); Imgs=list([]); L=0;
        img=cv.imread(Imgs0[0]); N1=np.shape(img)[0]; N2=np.shape(img)[1];
        L=Ls[2] 
    
        for img in range(len(Imgs0)):
            img=Imgs0[img];
            imgp=cv.cvtColor(cv.imread(img),cv.COLOR_BGR2RGB); imgp=imgp/256;
            lisp=np.zeros((0,3))
            for i in range(N1):
                for j in range(N2):
                    lisp=np.vstack([lisp,np.array([imgp[i,j,0],imgp[i,j,1],imgp[i,j,2]])])
            img0=np.zeros((0,3))
            for m in L:
                if m<0:
                    img0=np.vstack([img0,np.array([0,0,0])])
                else:
                    img0=np.vstack([img0,lisp[m]])
            Img0=np.zeros((N1,N2,3)); cou=-1
            for i in range(N1):
                for j in range(N2):
                    cou+=1;
                    Img0[i,j,0]=img0[cou,0]; Img0[i,j,1]=img0[cou,1]; Img0[i,j,2]=img0[cou,2]
            Imgs.append(Img0)
        
        for img in Imgs0:
            os.remove(img)
        
        os.chdir('../')
        os.rmdir("temp")
        for img in range(len(Imgs)):
            name=names[img]; img=Imgs[img];
            img*=256
            img.astype(int)
            cv.imwrite(name,img)
            
        with imageio.get_writer(Name+'.gif', mode='I',fps=FPS,loop=0) as writer:
            for name in names:
                image = imageio.imread(name)
                writer.append_data(image)
                os.remove(name)
        
        os.chdir('../')
                
    except Exception as e:
        print(f"Error: exception {e}.") 







def make_gif(Nimages,Name,Image,Resol,Shifts,Direction,FPS,Type,Mass,Charge,v):
    N1=Resol[0]; N2=Resol[1]; K1=Shifts[0]; K2=Shifts[1]; coe=Shifts[2];
    img1=np.zeros((N1,N2,3))
    for k in range(3):
        img1[:,:,k]=np.array([k*N1*N2+np.linspace(i*N2,(i+1)*N2-1,N2).astype(int) for i in range(N1)])
    img2=shadow4gif(Type,Mass,Charge,v,img1);
    lis1=np.zeros((0,3)); lis2=np.zeros((0,3))
    for i in range(N1):
        for j in range(N2):
            lis1=np.vstack([lis1,np.array([img1[i,j,0],img1[i,j,1],img1[i,j,2]])])
            lis2=np.vstack([lis2,np.array([img2[i,j,0],img2[i,j,1],img2[i,j,2]])])
    L=[];
    for k in range(N1*N2):
        pix=lis2[k,:]
        l=0;
        while (l<N1*N2 and (lis1[l,0]!=pix[0] or lis1[l,1]!=pix[1] or lis1[l,2]!=pix[2])):
            l+=1
        if l>=N1*N2:
            L.append(-1)
        else:
            L.append(l)
    Ls=list([N1,N2,L])

    
    img=cv.imread(Image); img=cv.cvtColor(img,cv.COLOR_BGR2RGB); img=img/256;
    N0=int(np.floor((np.shape(img)[0]-N1)/2)); names=[]; name0=Name; ext=".png"
    path=path0+'/'+Name+'_gif';
    try:
        os.mkdir(path)
        #print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    for j in range(Nimages):
        if j<10:
            names.append(name0+"000"+str(j)+ext)
        elif j<100:
            names.append(name0+"00"+str(j)+ext)
        elif j<1000:
            names.append(name0+"0"+str(j)+ext)
        else:
            names.append(name0+str(j)+ext)
    os.chdir(path)
    path=os.getcwd()+"/temp"
    try:
        os.mkdir(path)
        #print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    imgs=list([]);
    os.chdir(path)
    for i0 in range(len(names)):
        img0=np.zeros((N1,N2,3))
        floe1=int(np.floor(coe*i0)+K1); floe2=int(np.floor(coe*i0)+K2);
        for k in range(3):
            if Direction=="d2-":
                img0[:,:,k]=img[floe1:(floe1+N1),floe2:(floe2+N2),k]
            elif Direction=="d2+":
                img0[:,:,k]=img[(floe1+N1):floe1:-1,(floe2+N2):floe2:-1,k]
            elif Direction=="d1-":
                img0[:,:,k]=img[(floe1+N1):floe1:-1,floe2:(floe2+N2),k]
            elif Direction=="d1+":
                img0[:,:,k]=img[floe1:(floe1+N1),(floe2+N2):floe2:-1,k]
            elif Direction=="h-":
                img0[:,:,k]=img[N0+K1:N0+K1+N1,floe2:(floe2+N2),k]
            elif Direction=="h+":
                img0[:,:,k]=img[N0+K1+N1:N0+K1:-1,(floe2+N2):floe2:-1,k]
            elif Direction=="v+":
                img0[:,:,k]=img[floe1:(floe1+N1),N0+K2:N0+K2+N2,k]
            elif Direction=="v-":
                img0[:,:,k]=img[(floe1+N1):floe1:-1,N0+K2+N2:N0+K2:-1,k]
        imgs.append(img0)
        img0*=256
        img0.astype(int)
        cv.imwrite(names[i0],img0)
       
    Imgs0=os.listdir(); Imgs0.sort(); Imgs=list([]); L=0;
    img=cv.imread(Imgs0[0]); N1=np.shape(img)[0]; N2=np.shape(img)[1];
    L=Ls[2] 

    for img in range(len(Imgs0)):
        img=Imgs0[img];
        imgp=cv.cvtColor(cv.imread(img),cv.COLOR_BGR2RGB); imgp=imgp/256;
        lisp=np.zeros((0,3))
        for i in range(N1):
            for j in range(N2):
                lisp=np.vstack([lisp,np.array([imgp[i,j,0],imgp[i,j,1],imgp[i,j,2]])])
        img0=np.zeros((0,3))
        for m in L:
            if m<0:
                img0=np.vstack([img0,np.array([0,0,0])])
            else:
                img0=np.vstack([img0,lisp[m]])
        Img0=np.zeros((N1,N2,3)); cou=-1
        for i in range(N1):
            for j in range(N2):
                cou+=1;
                Img0[i,j,0]=img0[cou,0]; Img0[i,j,1]=img0[cou,1]; Img0[i,j,2]=img0[cou,2]
        Imgs.append(Img0)
    
    for img in Imgs0:
        os.remove(img)
    
    os.chdir('../')
    os.rmdir('temp')
    for img in range(len(Imgs)):
        name=names[img]; img=Imgs[img];
        img*=256
        img.astype(int)
        cv.imwrite(name,img)
    with imageio.get_writer(Name+'.gif', mode='I',fps=FPS,loop=0) as writer:
        for name in names:
            image = imageio.imread(name)
            writer.append_data(image)
            os.remove(name)

    os.chdir('../')


