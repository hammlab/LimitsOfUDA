import numpy as np
from scipy.optimize import minimize
from scipy import stats
from scipy import integrate


def pz(z,u,mu_p,mu_n,sig):
    u = np.array(u).reshape((2,1))
    return 0.5*stats.norm.pdf((z-np.dot(u.T,mu_p))/sig) +  0.5*stats.norm.pdf((z-np.dot(u.T,mu_n))/sig)

def fz(z,v,u,mu_p,mu_n,sig): # More general fz
    u = np.array(u).reshape((2,1))
    u /= np.sqrt(u[0]**2+u[1]**2)
    up = np.array([u[1],-u[0]]).reshape((2,1)) # 90 degress clockwise
    v = np.array(v).reshape((2,1))
    v /= np.sqrt(v[0]**2+v[1]**2)
    q = (v[1]*u[1]+v[0]*u[0])/(v[1]*u[0]-v[0]*u[1])*z
    if np.dot(up.T,v)<0:
        val = 0.5*stats.norm.cdf((q-np.dot(mu_p.T,up))/sig) + 0.5*stats.norm.cdf((q-np.dot(mu_n.T,up))/sig)
    elif np.dot(up.T,v)>0:
        val = 0.5*(1.-stats.norm.cdf((q-np.dot(mu_p.T,up))/sig)) + 0.5*(1.-stats.norm.cdf((q-np.dot(mu_n.T,up))/sig))
    else:
        if np.dot(u.T,v)>0:
            val = (1.+np.sign(z))/2
        else:
            val = (1.+np.sign(-z))/2
    return val


def err(v,u,a,b,mu_p,mu_n,sig):
    # int pz (h(z) - fz)^2 dz 
    u = np.array(u).reshape((2,1))
    def func(z,v,u,a,b,mu_p,mu_n,sig):
        return pz(z,u,mu_p,mu_n,sig)*((stats.norm.cdf(a*z+b)- fz(z,v,u,mu_p,mu_n,sig))**2) 
    res = integrate.quad(func,-np.inf,np.inf,args=(v,u,a,b,mu_p,mu_n,sig))
    return res[0]


def p_dist(u,mu_sp,mu_sn,mu_tp,mu_tn,sig):
    # int (psz - ptz)^2 dz
    def func(z,u,mu_sp,mu_sn,mu_tp,mu_tn,sig):
        return (pz(z,u,mu_sp,mu_sn,sig)-pz(z,u,mu_tp,mu_tn,sig))**2
    res = integrate.quad(func,-np.inf,np.inf,args=(u,mu_sp,mu_sn,mu_tp,mu_tn,sig))
    return res[0]


def fs_dist(vs,vt,u,mu_sp,mu_sn,mu_tp,mu_tn,sig):
    # es(fs,ft) = int psz (fsz - ftz)^2 dz
    u = np.array(u).reshape((2,1))
    def func(z,u,mu_sp,mu_sn,mu_tp,mu_tn,sig):
        return pz(z,u,mu_sp,mu_sn,sig)*((fz(z,vs,u,mu_sp,mu_sn,sig)- fz(z,vt,u,mu_tp,mu_tn,sig))**2) 
    res = integrate.quad(func,-np.inf,np.inf,args=(u,mu_sp,mu_sn,mu_tp,mu_tn,sig))
    return res[0]

def ft_dist(vs,vt,u,mu_sp,mu_sn,mu_tp,mu_tn,sig):
    # es(fs,ft) = int psz (fsz - ftz)^2 dz
    u = np.array(u).reshape((2,1))
    def func(z,u,mu_sp,mu_sn,mu_tp,mu_tn,sig):
        return pz(z,u,mu_tp,mu_tn,sig)*((fz(z,vs,u,mu_sp,mu_sn,sig)- fz(z,vt,u,mu_tp,mu_tn,sig))**2) 
    res = integrate.quad(func,-np.inf,np.inf,args=(u,mu_sp,mu_sn,mu_tp,mu_tn,sig))
    return res[0]


def loss(uab,mu_sp,mu_sn,mu_tp,mu_tn,sig,lamb=1E0,eta=1E1):
    u,a,b = [uab[0],uab[1]], uab[2], uab[3]
    return err(vs,u,a,b,mu_sp,mu_sn,sig)+ eta*((1.-(u[0]**2+u[1]**2))**2)\
         + lamb*p_dist(u,mu_sp,mu_sn,mu_tp,mu_tn,sig) 

##############################################################################

case = 3
if case==1: # favorable case
    mu_sp = np.array([-1.,1.]).reshape((2,1))
    mu_sn = np.array([-1.,-1.]).reshape((2,1))
    mu_tp = np.array([1.,1.]).reshape((2,1))
    mu_tn = np.array([1.,-1.]).reshape((2,1))
    vs = np.array([0.,1.]).reshape((2,1))
    vt = np.array([0.,1.]).reshape((2,1))
elif case==2: # unfavorable case
    mu_sp = np.array([-1.,1.]).reshape((2,1))
    mu_sn = np.array([-1.,-1.]).reshape((2,1))
    mu_tp = np.array([1.,-1.]).reshape((2,1))
    mu_tn = np.array([1.,1.]).reshape((2,1))
    vs = np.array([0.,1.]).reshape((2,1))
    vt = np.array([0.,-1.]).reshape((2,1))
else: # ambiguous case
    mu_sp = np.array([0.,1.]).reshape((2,1))
    mu_sn = np.array([0.,-1.]).reshape((2,1))
    mu_tp = np.array([-1.,0.]).reshape((2,1))
    mu_tn = np.array([1.,0.]).reshape((2,1))
    vs = np.array([0.,1.]).reshape((2,1))
    vt = np.array([-1.,0.]).reshape((2,1))

## Repeat the following to get statistics
#u0 = [np.sqrt(0.5),np.sqrt(0.5)]
#u0 = [0.,1.]
u0 = np.random.normal(size=(2,1))
u0[1] = np.abs(u0[1]) 
u0 /= np.sqrt(u0[0]**2+u0[1]**2)
print(u0)
a0 = 1.
b0 = 0.
uab0 = np.array([u0[0],u0[1],a0,b0]).reshape((4,1))
sig = 1.
lamb = 1E-2
eta = 1E1

## Solve
res = minimize(loss,uab0,args=(mu_sp,mu_sn,mu_tp,mu_tn,sig,lamb,eta),method='Nelder-Mead',options={'disp':True,'maxiter':1000})
u,a,b = res.x[0:2],res.x[2],res.x[3]

print('u0=',u0)
print('u=',u)
print(a,b)
print('es=',err(vs,u,a,b,mu_sp,mu_sn,sig))
print('et=',err(vt,u,a,b,mu_tp,mu_tn,sig))
print('eta*res=',eta*((1.-(u[0]**2+u[1]**2))**2))
print('p_dist=',p_dist(u,mu_sp,mu_sn,mu_tp,mu_tn,sig))
print('fs_dist=',fs_dist(vs,vt,u,mu_sp,mu_sn,mu_tp,mu_tn,sig))
print('ft_dist=',ft_dist(vs,vt,u,mu_sp,mu_sn,mu_tp,mu_tn,sig))