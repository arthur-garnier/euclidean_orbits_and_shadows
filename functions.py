#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 18:02:59 2024

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

def orbit_ERN(Mass,Charge,Tau,N,IniConds,lim):
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
    R=CC[0]; Rp=CC[1]; P=CC[4]; Pp=CC[5];# T=CC[2]; Tp=CC[3];
    #J=R**2*Pp; E=R**2*Pp**2+R*Rp**2/(R-2)+1-2/R; E=1/np.sqrt(E); L=J*E; g2=1/12+1/L**2; g3=1/216-(2-3*(1-2/R)**2*E**2)/(12*L**2);
    J=R**2*Pp; H=R**2*Pp**2+R**2*Rp**2/(R**2-2*R+Q**2)+1-2/R+Q**2/R**2; E=(1-2/R+Q**2/R**2)/np.sqrt(H); L=J/np.sqrt(H);
    rpol=np.roots([(1-E**2)/L**2,-2/L**2,Q**2/L**2-1,2,-Q**2]);
    mi=min(abs(rpol-np.real(rpol))); frpol=[np.real(rpol[rr]) for rr in np.where(abs(rpol-np.real(rpol))==mi)][0];
    rbar=frpol[np.where(abs(frpol)==min(abs(frpol)))[0][0]];
    delta=(1-E**2)/L**2; gamma=2*(2*rbar*delta-1/L**2); beta=6*delta*rbar**2-6*rbar/L**2+Q**2/L**2-1; alpha=2*(2*delta*rbar**3-3*rbar**2/L**2+(Q**2/L**2-1)*rbar+1);
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
        return((2-rbar)*(4*np.real(weierP(Z0+t))-beta/3)-alpha);#return(np.real(weierP(Z0+t))-1/6)
    
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
    plt.title("Orbit with M="+str(Mass)+", Q="+str(Q)+" and E^2="+str(E**2))
    plt.show()











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
    H=Q**2*(eps*D+D**(-1)*Rp**2/R**4+Pp**2); E=D/np.sqrt(H);
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
    R=Vecc[:,0]; theta=Vecc[:,1]; phi=Vecc[:,2];
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    
    x = (m+np.sqrt(m**2-q**2)) * np.outer(np.cos(u), np.sin(v))
    y = (m+np.sqrt(m**2-q**2)) * np.outer(np.sin(u), np.sin(v))
    z = (m+np.sqrt(m**2-q**2)) * np.outer(np.ones(np.size(u)), np.cos(v))
    
    #ax.plot_surface(x, y, z)
    ax.plot(R*np.sin(theta)*np.cos(phi),R*np.sin(theta)*np.sin(phi),R*np.cos(theta));
    
    ax.set_aspect('equal')
    plt.title("Orbit with eps="+str(eps)+", Q="+str(Q)+", m="+str(m)+", q="+str(q)+" and E^2="+str(np.floor(1e6*E**2)*1e-6))
    plt.show()
    
    
    









def shadow_ERN(Mass,Charge,v,Image):
    x0=7; rf=10; Xmax=2.3; Q=Charge/Mass;
    Img=cv.imread(Image);
    #Img=Image;
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
        r=Z[0]; rp=Z[2]; php=Z[3];#ph=Z[1];
        co=np.sqrt((Q**2*php**2 + php**2*r**2 - 2*php**2*r + rp**2)*((v**2 + 1)*Q**2 + r**2*v**2 + (-2*v**2 - 2)*r))*(Q**2 + r**2 - 2*r)/(((Q**2 + r**2 - 2*r)*php**2 + rp**2)*r**3)
        Z[2]=co*Z[2]; Z[3]=co*Z[3];
        #Z=To_spherical(x0,co*h/np.sqrt(h**2+y**2),y,-co*y/np.sqrt(h**2+y**2));
        #Z=[Z[0]/Mass,Z[2],Z[1],Z[3]*Mass];
        return(Z)
    
    def weierP(g2,g3,z):
        N0=12;
        zz0=z/(2**N0); zz=1/zz0**2+g2/20*zz0**2+g3/28*zz0**4;
        for j in range(N0):
            zz=-2*zz+(6*zz**2-g2/2)**2/(4*(4*zz**3-g2*zz-g3));
        return(zz)
    
    def newton(g2,g3,Z,t):
        def toanihil(s):
            #return((rf/(2*Mass))*(4*np.real(weierP(g2,g3,Z+s))+1/3)-1);
            return((rf/Mass-rbar)*(4*np.real(weierP(g2,g3,Z+s))-beta/3)-alpha);
        sgn=np.sign(toanihil(t)); sol=t; step=-0.1;
        while sgn*toanihil(sol)>0:
            sol=sol+step;
        #epss=1e-12; itermax=100; iter=0; #sol2=sol+1e-5;
        #while (abs(toanihil(sol))>epss and iter<itermax):
            #solp=sol2; sol2=sol2-(sol2-sol)*toanihil(sol2)/(toanihil(sol2)-toanihil(sol)); sol=solp; iter+=1;
        #    sol=sol-toanihil(sol)/derivative(toanihil,sol,dx=1e-3); iter+=1;
        sol=nt(toanihil,sol);
        return(sol)
    
    Xred=np.zeros((0,2));
    AR=[];
    for xx in XX[int(np.floor(Npix/2)):Npix]:
        for yy in YY[int(np.floor(Npiy/2)):Npiy]:
            AR.append(np.sqrt(xx**2+yy**2));
    AR=np.sort(AR)
    for zz in [zu for zu in AR if zu!=0]:#AR:
        X=init_conds(zz)
        r=X[0]; ph=X[1]; rp=X[2]; php=X[3];
        #J=r**2*php; E=r**2*php**2+r*rp**2/(r-2)+1-2/r; E=1/np.sqrt(E); L=J*E; g2=1/12+1/L**2; g3=1/216-(2-3*(1-2/r)**2*E**2)/(12*L**2);
        J=r**2*php; H=r**2*php**2+r**2*rp**2/(r**2-2*r+Q**2)+1-2/r+Q**2/r**2; E=(1-2/r+Q**2/r**2)/np.sqrt(H); L=J/np.sqrt(H);
        rpol=np.roots([(1-E**2)/L**2,-2/L**2,Q**2/L**2-1,2,-Q**2]);
        mi=min(abs(rpol-np.real(rpol))); frpol=[np.real(rpol[rr]) for rr in np.where(abs(rpol-np.real(rpol))==mi)][0];
        rbar=frpol[np.where(abs(frpol)==min(abs(frpol)))[0][0]];
        delta=(1-E**2)/L**2; gamma=2*(2*rbar*delta-1/L**2); beta=6*delta*rbar**2-6*rbar/L**2+Q**2/L**2-1; alpha=2*(2*delta*rbar**3-3*rbar**2/L**2+(Q**2/L**2-1)*rbar+1);
        g2=(beta**2/3-alpha*gamma)/4; g3=(alpha*beta*gamma/6-alpha**2*delta/2-beta**3/27)/8;
        rp2=np.roots([4,0,-g2,-g3]); z0=alpha/(4*(r-rbar))+beta/12;#z0=1/(2*r)-1/12;
        if abs(rp)<1e-12:
           Z0=carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2])
        else:
           Z0=np.sign(-rp)*carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2]);
        new=newton(g2,g3,Z0,0); P=ph+np.sign(php)*new;
        Xred=np.vstack([Xred,np.array([zz,P])]);
        #def wrs(t):
        #    return(np.real((router-rbar)*(4*weierP(g2,g3,Z0+t)-beta/3)-alpha));#
        #xxx=fsolve(wrs,0,full_output=False)[0]; vvv=wrs(xxx);
        #if (abs(vvv)>=1e-8 or abs(np.sign(php)*xxx+ph)>=2*np.pi):#P>np.pi/2*(1+np.sign(1-Q**2))/2:
        #    Xred=np.vstack([Xred,np.array([zz,P])]);
    
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
                #if (t1<0 or t1>1 or t2<0 or t2>1):
                #    xred[i,j,0]=0; xred[i,j,1]=0; xred[i,j,2]=0;
                
    xredt=np.zeros((Npiy,Npix,3));
    for k in range(3):
        xredt[:,:,k]=np.transpose(xred[:,:,k]);
    
    plt.figure(figsize=(12,12))
    plt.imshow(xredt)
    plt.grid(False)
    plt.axis('off')
    plt.show()





def shadow_LRN(Mass,Charge,v,Image):
    x0=7; rf=10; Xmax=2.3; Q=Charge/Mass;
    if v<1:
        mu=-1
    elif v==1:
        mu=0
    else:
        mu=1
    Img=cv.imread(Image);
    #Img=Image;
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
        #th=-np.arctan2(-y,h);
        Z=[Z[0]/Mass,Z[2],Z[1],Z[3]*Mass];
        r=Z[0]; rp=Z[2]; php=Z[3];#ph=Z[1];
        if mu!=0:
            co=np.sqrt((Q**2*php**2 + php**2*r**2 - 2*php**2*r + rp**2)*((-v**2 + 1)*Q**2 + 2*r**2-r**2*v**2 + (2*v**2 - 2)*r))*(Q**2 + r**2 - 2*r)/(((Q**2 + r**2 - 2*r)*php**2 + rp**2)*r**3)
        else:
            co=r*np.sqrt((Q**2*php**2 + php**2*r**2 - 2*php**2*r + rp**2))*(Q**2 + r**2 - 2*r)/(((Q**2 + r**2 - 2*r)*php**2 + rp**2)*r**3)
        Z[2]=co*Z[2]; Z[3]=co*Z[3];
        #Z=To_spherical(x0,co*h/np.sqrt(h**2+y**2),y,-co*y/np.sqrt(h**2+y**2));
        #Z=[Z[0]/Mass,Z[2],Z[1],Z[3]*Mass];
        return(Z)
    
    def weierP(g2,g3,z):
        N0=12;
        zz0=z/(2**N0); zz=1/zz0**2+g2/20*zz0**2+g3/28*zz0**4;
        for j in range(N0):
            zz=-2*zz+(6*zz**2-g2/2)**2/(4*(4*zz**3-g2*zz-g3));
        return(zz)
    
    def newton(g2,g3,Z,t):
        def toanihil(s):
            #return((rf/(2*Mass))*(4*np.real(weierP(g2,g3,Z+s))+1/3)-1);
            return((rf/Mass-rbar)*(4*np.real(weierP(g2,g3,Z+s))-beta/3)-alpha);
        sgn=np.sign(toanihil(t)); sol=t; step=-0.05;
        while sgn*toanihil(sol)>0:
            sol=sol+step;
        #epss=1e-12; itermax=100; iter=0; #sol2=sol+1e-5;
        #while (abs(toanihil(sol))>epss and iter<itermax):
            #solp=sol2; sol2=sol2-(sol2-sol)*toanihil(sol2)/(toanihil(sol2)-toanihil(sol)); sol=solp; iter+=1;
        #    sol=sol-toanihil(sol)/derivative(toanihil,sol,dx=1e-3); iter+=1;
        sol=nt(toanihil,sol);
        return(sol)
    
    Xred=np.zeros((0,2));
    AR=[];
    for xx in XX[int(np.floor(Npix/2)):Npix]:
        for yy in YY[int(np.floor(Npiy/2)):Npiy]:
            AR.append(np.sqrt(xx**2+yy**2));
    AR=np.sort(AR)
    for zz in [zu for zu in AR if zu!=0]:#AR:
        X=init_conds(zz)
        r=X[0]; ph=X[1]; rp=X[2]; php=X[3];
        #J=r**2*php; E=r**2*php**2+r*rp**2/(r-2)+1-2/r; E=1/np.sqrt(E); L=J*E; g2=1/12+1/L**2; g3=1/216-(2-3*(1-2/r)**2*E**2)/(12*L**2);
        if mu!=0:
            J=r**2*php; H=r**2*php**2+r**2*rp**2/(r**2-2*r+Q**2)-(1-2/r+Q**2/r**2); E=(1-2/r+Q**2/r**2)/np.sqrt(H); L=J/np.sqrt(H);
        else:
            E=1;
            #L=r**2*np.sin(th)/np.sqrt((r**2+Q**2-2*r));
            L=r**2*php/(1-2/r+Q**2/r**2)
        rpol=np.roots([(E**2+mu)/L**2,-2*mu/L**2,-1+mu*Q**2/L**2,2,-Q**2]);
        mi=min(abs(rpol-np.real(rpol))); frpol=[np.real(rpol[rr]) for rr in np.where(abs(rpol-np.real(rpol))==mi)][0];
        rbar=frpol[np.where(abs(frpol)==min(abs(frpol)))[0][0]];
        delta=(E**2+mu)/L**2; gamma=2*(2*delta*rbar-mu/L**2); beta=-1+mu*Q**2/L**2+3*rbar*(gamma-2*delta*rbar); alpha=2+rbar*(2*beta-rbar*(3*gamma-4*delta*rbar));
        g2=(beta**2/3-alpha*gamma)/4; g3=(alpha*beta*gamma/6-alpha**2*delta/2-beta**3/27)/8;
        rp2=np.roots([4,0,-g2,-g3]); z0=alpha/(4*(r-rbar))+beta/12;#z0=1/(2*r)-1/12;
        if abs(rp)<1e-12:
           Z0=carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2])
        else:
           Z0=np.sign(-rp)*carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2]);
        new=newton(g2,g3,Z0,0); P=ph+np.sign(php)*new;
        if P>np.pi/2:
            Xred=np.vstack([Xred,np.array([zz,P])]);
        #def wrs(t):
        #    return(np.real((router-rbar)*(4*weierP(g2,g3,Z0+t)-beta/3)-alpha));#
        #xxx=fsolve(wrs,0,full_output=False)[0]; vvv=wrs(xxx);
        #if (abs(vvv)>=1e-8 or abs(np.sign(php)*xxx+ph)>=2*np.pi):#P>np.pi/2*(1+np.sign(1-Q**2))/2:
        #    Xred=np.vstack([Xred,np.array([zz,P])]);
    
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
                #if (t1<0 or t1>1 or t2<0 or t2>1):
                #    xred[i,j,0]=0; xred[i,j,1]=0; xred[i,j,2]=0;
                
    xredt=np.zeros((Npiy,Npix,3));
    for k in range(3):
        xredt[:,:,k]=np.transpose(xred[:,:,k]);
    
    plt.figure(figsize=(12,12))
    plt.imshow(xredt)
    plt.grid(False)
    plt.axis('off')
    plt.show()








def deflection_ERN(Mass,Charge,v,N):
    Q=Charge/Mass;
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
        E=1/np.sqrt(v**2+1)
        #b=r0*np.sin(th)*np.sqrt(1+2/(v**2*(2-r0)))
        b=r0*np.sin(th)/v*np.sqrt((Q**2*(1+v**2)+r0*v**2*(r0-2)-2*r0)/(Q**2+r0**2-2*r0))
        Z=[r0,b,E,th];#Z=[Z[0]/Mass,Z[2],Z[1],Z[3]*Mass];
        return(Z)    
    
    Xred=np.zeros((0,2));
    AR=[];
    for xx in XX[int(np.floor(Npix/2)):Npix]:
        for yy in YY[int(np.floor(Npiy/2)):Npiy]:
            AR.append(np.sqrt(xx**2+yy**2));
    AR=np.sort(AR)
    for zz in [zu for zu in AR if zu!=0]:
        X=init_conds(zz); r=X[0]; b=X[1]; E=X[2]; th=X[3]; L=b*v*E;
        rpol=np.roots([(1-E**2)/L**2,-2/L**2,Q**2/L**2-1,2,-Q**2]);
        mi=min(abs(rpol-np.real(rpol))); frpol=[np.real(rpol[rr]) for rr in np.where(abs(rpol-np.real(rpol))==mi)][0];
        rbar=frpol[np.where(abs(frpol)==min(abs(frpol)))[0][0]];
        delta=(1-E**2)/L**2; gamma=2*(2*rbar*delta-1/L**2); beta=6*delta*rbar**2-6*rbar/L**2+Q**2/L**2-1; alpha=2*(2*delta*rbar**3-3*rbar**2/L**2+(Q**2/L**2-1)*rbar+1);
        g2=(beta**2/3-alpha*gamma)/4; g3=(alpha*beta*gamma/6-alpha**2*delta/2-beta**3/27)/8;
        rp2=np.roots([4,0,-g2,-g3]);
        rmin=max(rpol);
        #g2=1/12+1/L**2; g3=1/216-(2-3*E**2)/(12*L**2);
        #rp2=np.roots([4,0,-g2,-g3]); #z0=alpha/(4*(r-rbar))+beta/12;#z0=1/(2*r)-1/12;
        #rpol=np.roots([(1-E**2)/L**2,-2/L**2,-1,2]);
        #rmin=max(rpol)
        zmin=alpha/(4*(rmin-rbar))+beta/12; zinf=beta/12; z0=alpha/(4*(r-rbar))+beta/12;
        #Rf1=carlson(-1/12-rp2[0],-1/12-rp2[1],-1/12-rp2[2])-carlson(-1/12-rp2[0]+1/(2*rmin),-1/12-rp2[1]+1/(2*rmin),-1/12-rp2[2]+1/(2*rmin));
        #Rf2=carlson(1/(2*r)-1/12-rp2[0],1/(2*r)-1/12-rp2[1],1/(2*r)-1/12-rp2[2])-carlson(-1/12-rp2[0]+1/(2*rmin),-1/12-rp2[1]+1/(2*rmin),-1/12-rp2[2]+1/(2*rmin));
        Rf1=carlson(zinf-rp2[0],zinf-rp2[1],zinf-rp2[2])-carlson(zmin-rp2[0],zmin-rp2[1],zmin-rp2[2]);
        Rf2=carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2])-carlson(zmin-rp2[0],zmin-rp2[1],zmin-rp2[2]);
        Xred=np.vstack([Xred,np.array([zz,np.real(-(Rf1+Rf2)+th-np.pi)])]);#Xred=np.vstack([Xred,np.array([zz,(-2*(Rf1)-np.pi)])]);
    
    cmap=plt.get_cmap('RdBu_r',len(Xred[:,1]))#RdBu_r
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
    MD=max(abs(md),abs(Md));#norm = mpl.colors.Normalize(vmin=md, vmax=Md)
    norm = mpl.colors.Normalize(vmin=-MD, vmax=MD)
    for i in range(Npix):
        xx=XX[i];
        for j in range(Npiy):
            yy=YY[j];
            if KK[i,j]!=0:
                coe=Xred[int(KK[i,j]),1]; xred0=cmap(int((coe+MD)/(2*MD)*len(Xred[:,1])));
                xred[i,j,:]=[xred0[0],xred0[1],xred0[2]]
                
    xredt=np.zeros((Npiy,Npix,3));
    for k in range(3):
        xredt[:,:,k]=np.transpose(xred[:,:,k]);
    
    plt.figure(figsize=(12,12))
    plt.imshow(xredt)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
    sm.set_array([])
    plt.colorbar(sm, ticks=np.linspace(-MD,MD, 11+10),label='$\Delta\phi[rad]$',shrink=0.812) 
    plt.grid(False)
    plt.axis('off')
    plt.legend()
    plt.show()









def deflection_LRN(Mass,Charge,v,N):
    Q=Charge/Mass;
    if v<1:
        mu=-1
    elif v==1:
        mu=0
    else:
        mu=1
    Mass=0.35*Mass; r0=100; Xmax=2.3/Mass; Npix=N; Npiy=N; r0=r0/Mass; h=r0/2;
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
        if mu!=0:
            E=1/np.sqrt(1-v**2)
            #b=r0*np.sin(th)*np.sqrt(1+2/(v**2*(r0-2)))
            b=r0*np.sin(th)*np.sqrt((Q**2*mu*v**2 + r0**2*mu*v**2 - 2*r0*mu*v**2 - Q**2*mu - r0**2*mu - r0**2 + 2*r0*mu)/((Q**2 + r0**2 - 2*r0)*(mu*v**2-mu-1)))
            L=b*v*E;
        else:
            E=1
            #b=r0*np.sin(th)*np.sqrt(1+2/(r0-2))
            b=r0**2*np.sin(th)/np.sqrt((r0**2+Q**2-2*r0));
            L=b*E
        Z=[r0,E,L,th];#Z=[Z[0]/Mass,Z[2],Z[1],Z[3]*Mass];
        return(Z)    
    
    Xred=np.zeros((0,2));
    AR=[];
    for xx in XX[int(np.floor(Npix/2)):Npix]:
        for yy in YY[int(np.floor(Npiy/2)):Npiy]:
            AR.append(np.sqrt(xx**2+yy**2));
    AR=np.sort(AR)
    for zz in [zu for zu in AR if zu!=0]:
        X=init_conds(zz); r=X[0]; E=X[1]; L=X[2]; th=X[3];
        rpol=np.roots([(E**2+mu)/L**2,-2*mu/L**2,-1+mu*Q**2/L**2,2,-Q**2]);
        mi=min(abs(rpol-np.real(rpol))); frpol=[np.real(rpol[rr]) for rr in np.where(abs(rpol-np.real(rpol))==mi)][0];
        rbar=frpol[np.where(abs(frpol)==min(abs(frpol)))[0][0]];
        delta=(E**2+mu)/L**2; gamma=2*(2*delta*rbar-mu/L**2); beta=-1+mu*Q**2/L**2+3*rbar*(gamma-2*delta*rbar); alpha=2+rbar*(2*beta-rbar*(3*gamma-4*delta*rbar));
        g2=(beta**2/3-alpha*gamma)/4; g3=(alpha*beta*gamma/6-alpha**2*delta/2-beta**3/27)/8;
        rp2=np.roots([4,0,-g2,-g3]);
        rmin=max(rpol);
        #g2=1/12+mu/L**2; g3=1/216-1/(4*b**2)-mu/(6*L**2);
        #rp2=np.roots([4,0,-g2,-g3]); #z0=alpha/(4*(r-rbar))+beta/12;#z0=1/(2*r)-1/12;
        #rpol=np.roots([1/b**2+mu/L**2,-2*mu/L**2,-1,2]);
        #rmin=max(rpol)
        zmin=alpha/(4*(rmin-rbar))+beta/12; zinf=beta/12; z0=alpha/(4*(r-rbar))+beta/12;
        #Rf1=carlson(-1/12-rp2[0],-1/12-rp2[1],-1/12-rp2[2])-carlson(-1/12-rp2[0]+1/(2*rmin),-1/12-rp2[1]+1/(2*rmin),-1/12-rp2[2]+1/(2*rmin));
        #Rf2=carlson(1/(2*r)-1/12-rp2[0],1/(2*r)-1/12-rp2[1],1/(2*r)-1/12-rp2[2])-carlson(-1/12-rp2[0]+1/(2*rmin),-1/12-rp2[1]+1/(2*rmin),-1/12-rp2[2]+1/(2*rmin));
        Rf1=carlson(zinf-rp2[0],zinf-rp2[1],zinf-rp2[2])-carlson(zmin-rp2[0],zmin-rp2[1],zmin-rp2[2]);
        Rf2=carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2])-carlson(zmin-rp2[0],zmin-rp2[1],zmin-rp2[2]);
        if np.real(-(Rf1+Rf2)+th-np.pi)>0:
            Xred=np.vstack([Xred,np.array([zz,np.real(-(Rf1+Rf2)+th-np.pi)])]);#Xred=np.vstack([Xred,np.array([zz,(-2*(Rf1)-np.pi)])]);
    
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
    MD=max(abs(md),abs(Md));#norm = mpl.colors.Normalize(vmin=md, vmax=Md)
    norm = mpl.colors.Normalize(vmin=-0*MD, vmax=MD)
    for i in range(Npix):
        xx=XX[i];
        for j in range(Npiy):
            yy=YY[j];
            if KK[i,j]!=0:
                coe=Xred[int(KK[i,j]),1]; xred0=cmap(int((coe+0*MD)/(1*MD)*len(Xred[:,1])));
                xred[i,j,:]=[xred0[0],xred0[1],xred0[2]]
                
    xredt=np.zeros((Npiy,Npix,3));
    for k in range(3):
        xredt[:,:,k]=np.transpose(xred[:,:,k]);
    
    plt.figure(figsize=(12,12))
    plt.imshow(xredt)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
    sm.set_array([])
    plt.colorbar(sm, ticks=np.linspace(-0*MD,MD, 11+10),label='$\Delta\phi[rad]$',shrink=0.812) 
    plt.grid(False)
    plt.axis('off')
    plt.legend()
    plt.show()