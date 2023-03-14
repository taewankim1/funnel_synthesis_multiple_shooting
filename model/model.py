
# coding: utf-8

# In[ ]:

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import time
import random
import IPython
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))

class ODE_solution(object) :
    def __init__(self) :
        pass
    def setter(self,y,t) :
        self.y = y
        self.t = t

def RK4(odefun, tspan, y0,args,N_RK=10) :
    t = np.linspace(tspan[0],tspan[-1],N_RK)
    h = t[1] - t[0]
    iy = len(y0)
    y_sol = np.zeros((N_RK,iy))
    y_sol[0] = y0
    for idx in range(0,N_RK-1) :
        tk = t[idx]
        yk = y_sol[idx]
        k1 = odefun(tk,yk,*args)
        k2 = odefun(tk + h/2,yk + h/2*k1,*args)
        k3 = odefun(tk + h/2,yk + h/2*k2,*args)
        k4 = odefun(tk+h,yk+h*k3,*args)
        y_sol[idx+1] = yk + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    sol = ODE_solution()
    sol.setter(y_sol.T,t)
    return sol


class OptimalcontrolModel(object) :
    def __init__(self,name,ix,iu,linearization) :
        self.name = name
        self.ix = ix
        self.iu = iu
        self.type_linearization = linearization

    def forward(self,x,u,idx=None):
        print("this is in parent class")
        pass

    def diff(self,x,u) :
        if self.type_linearization == "numeric_central" :
            A,B = self.diff_numeric_central(x,u)
        elif self.type_linearization == "numeric_forward" :
            A,B = self.diff_numeric(x,u)
        elif self.type_linearization == "analytic" :
            A,B = self.diff_analytic(x,u)
        return A,B

    def diff_analytic(self,x,u) :
        print("this is in parent class")
        pass

    def diff_numeric_central(self,x,u) :
        # state & input size
        ix = self.ix
        iu = self.iu
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
        
        # numerical difference
        h = pow(2,-17) / 2 
        eps_x = np.identity(ix)
        eps_u = np.identity(iu)

        # expand to tensor
        x_mat = np.expand_dims(x,axis=2)
        u_mat = np.expand_dims(u,axis=2)

        # diag
        x_diag = np.tile(x_mat,(1,1,ix))
        u_diag = np.tile(u_mat,(1,1,iu))

        # augmented = [x_aug x], [u, u_aug]
        x_aug_m = x_diag - eps_x * h
        x_aug_m = np.dstack((x_aug_m,np.tile(x_mat,(1,1,iu))))
        x_aug_m = np.reshape( np.transpose(x_aug_m,(0,2,1)), (N*(iu+ix),ix))

        u_aug_m = u_diag - eps_u * h
        u_aug_m = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug_m))
        u_aug_m = np.reshape( np.transpose(u_aug_m,(0,2,1)), (N*(iu+ix),iu))

        # augmented = [x_aug x], [u, u_aug]
        x_aug_p = x_diag + eps_x * h
        x_aug_p = np.dstack((x_aug_p,np.tile(x_mat,(1,1,iu))))
        x_aug_p = np.reshape( np.transpose(x_aug_p,(0,2,1)), (N*(iu+ix),ix))

        u_aug_p = u_diag + eps_u * h
        u_aug_p = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug_p))
        u_aug_p = np.reshape( np.transpose(u_aug_p,(0,2,1)), (N*(iu+ix),iu))

        # numerical difference
        f_change_m = self.forward(x_aug_m,u_aug_m,0)
        f_change_p = self.forward(x_aug_p,u_aug_p,0)
        f_change_m = np.reshape(f_change_m,(N,ix+iu,ix))
        f_change_p = np.reshape(f_change_p,(N,ix+iu,ix))
        f_diff = (f_change_p - f_change_m) / (2*h)
        f_diff = np.transpose(f_diff,[0,2,1])
        fx = f_diff[:,:,0:ix]
        fu = f_diff[:,:,ix:ix+iu]
        
        return np.squeeze(fx), np.squeeze(fu)

    def diff_numeric(self,x,u) :
        # state & input size
        ix = self.ix
        iu = self.iu
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
        
        # numerical difference
        h = pow(2,-17)
        eps_x = np.identity(ix)
        eps_u = np.identity(iu)

        # expand to tensor
        x_mat = np.expand_dims(x,axis=2)
        u_mat = np.expand_dims(u,axis=2)

        # diag
        x_diag = np.tile(x_mat,(1,1,ix))
        u_diag = np.tile(u_mat,(1,1,iu))

        # augmented = [x_aug x], [u, u_aug]
        x_aug_m = x_diag - eps_x * 0
        x_aug_m = np.dstack((x_aug_m,np.tile(x_mat,(1,1,iu))))
        x_aug_m = np.reshape( np.transpose(x_aug_m,(0,2,1)), (N*(iu+ix),ix))

        u_aug_m = u_diag - eps_u * 0
        u_aug_m = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug_m))
        u_aug_m = np.reshape( np.transpose(u_aug_m,(0,2,1)), (N*(iu+ix),iu))

        # augmented = [x_aug x], [u, u_aug]
        x_aug_p = x_diag + eps_x * h
        x_aug_p = np.dstack((x_aug_p,np.tile(x_mat,(1,1,iu))))
        x_aug_p = np.reshape( np.transpose(x_aug_p,(0,2,1)), (N*(iu+ix),ix))

        u_aug_p = u_diag + eps_u * h
        u_aug_p = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug_p))
        u_aug_p = np.reshape( np.transpose(u_aug_p,(0,2,1)), (N*(iu+ix),iu))

        # numerical difference
        f_change_m = self.forward(x_aug_m,u_aug_m,0)
        f_change_p = self.forward(x_aug_p,u_aug_p,0)
        f_change_m = np.reshape(f_change_m,(N,ix+iu,ix))
        f_change_p = np.reshape(f_change_p,(N,ix+iu,ix))
        f_diff = (f_change_p - f_change_m) / (h)
        f_diff = np.transpose(f_diff,[0,2,1])
        fx = f_diff[:,:,0:ix]
        fu = f_diff[:,:,ix:ix+iu]
        
        return np.squeeze(fx), np.squeeze(fu)

    def diff_discrete_zoh(self,x,u,delT,tf) :
        # delT = self.delT
        ix = self.ix
        iu = self.iu

        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)

        def dvdt(t,V,u,length) :
            assert len(u) == length
            V = V.reshape((length,ix + ix*ix + ix*iu + ix + ix)).transpose()
            x = V[:ix].transpose()
            Phi = V[ix:ix*ix + ix]
            Phi = Phi.transpose().reshape((length,ix,ix))
            Phi_inv = np.linalg.inv(Phi)
            f = self.forward(x,u)
            # if self.type_linearization == "numeric_central" :
            #     A,B = self.diff_numeric_central(x,u)
            # elif self.type_linearization == "numeric_forward" :
            #     A,B = self.diff_numeric(x,u)
            # elif self.type_linearization == "analytic" :
            A,B = self.diff(x,u)
            # IPython.embed()
            dpdt = np.matmul(A,Phi).reshape((length,ix*ix)).transpose()
            dbdt = np.matmul(Phi_inv,B).reshape((length,ix*iu)).transpose()
            dsdt = np.squeeze(np.matmul(Phi_inv,np.expand_dims(f,2))).transpose() / tf
            dzdt = np.squeeze(np.matmul(Phi_inv,-np.matmul(A,np.expand_dims(x,2)) - np.matmul(B,np.expand_dims(u,2)))).transpose()
            dv = np.vstack((f.transpose(),dpdt,dbdt,dsdt,dzdt))
            return dv.flatten(order='F')
        
        A0 = np.eye(ix).flatten()
        B0 = np.zeros((ix*iu))
        s0 = np.zeros(ix)
        z0 = np.zeros(ix)
        V0 = np.array([np.hstack((x[i],A0,B0,s0,z0)) for i in range(N)]).transpose()
        V0_repeat = V0.flatten(order='F')

        sol = solve_ivp(dvdt,(0,delT),V0_repeat,args=(u,N),method='RK45',rtol=1e-6,atol=1e-10)
        # sol = solve_ivp(dvdt,(0,delT),V0_repeat,args=(u,N),method='RK45',max_step=1e-2,rtol=1e-6,atol=1e-10)
        # sol = solve_ivp(dvdt,(0,delT),V0_repeat,args=(u,N),method='RK45',max_step=1e-2)
        # IPython.embed()
        idx_state = slice(0,ix)
        idx_A = slice(ix,ix+ix*ix)
        idx_B = slice(ix+ix*ix,ix+ix*ix+ix*iu)
        idx_s = slice(ix+ix*ix+ix*iu,ix+ix*ix+ix*iu+ix)
        idx_z = slice(ix+ix*ix+ix*iu+ix,ix+ix*ix+ix*iu+ix+ix)
        sol = sol.y[:,-1].reshape((N,-1))
        xnew = np.zeros((N+1,ix))
        xnew[0] = x[0]
        xnew[1:] = sol[:,:ix]
        x_prop = sol[:,idx_state].reshape((-1,ix))
        A = sol[:,idx_A].reshape((-1,ix,ix))
        B = np.matmul(A,sol[:,idx_B].reshape((-1,ix,iu)))
        s = np.matmul(A,sol[:,idx_s].reshape((-1,ix,1))).squeeze()
        z = np.matmul(A,sol[:,idx_z].reshape((-1,ix,1))).squeeze()

        return A,B,s,z,x_prop
 
    def diff_discrete_foh_variational(self,x,u,dtau,tf) :
        ix = self.ix
        iu = self.iu

        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
        idx_state = slice(0,ix)
        idx_A = slice(ix,ix+ix*ix)
        idx_Bm = slice(ix+ix*ix,ix+ix*ix+ix*iu)
        idx_Bp = slice(ix+ix*ix+ix*iu,ix+ix*ix+2*ix*iu)
        idx_s = slice(ix+ix*ix+2*ix*iu,ix+ix*ix+2*ix*iu+ix)
        # idx_z = slice(ix+ix*ix+2*ix*iu+ix,ix+ix*ix+2*ix*iu+ix+ix)
        def dvdt(t,V,um,up,length) :
            assert len(um) == len(up)
            assert len(um) == length
            alpha = (dtau - t) / dtau
            beta = t / dtau
            u = alpha * um + beta * up
            # V = V.reshape((length,ix + ix*ix + 2*ix*iu + ix + ix)).transpose()
            V = V.reshape((length,ix + ix*ix + 2*ix*iu + ix)).transpose()
            x = V[:ix].transpose()
            Phi = V[ix:ix*ix + ix]
            Phi = Phi.transpose().reshape((length,ix,ix))
            x3 = V[idx_Bm].transpose().reshape(length,ix,iu)
            x4 = V[idx_Bp].transpose().reshape(length,ix,iu)
            x5 = V[idx_s].transpose().reshape(length,ix,1)
            # x6 = V[idx_z].transpose().reshape(length,ix,1)
            f = self.forward(x,u)
            # if self.type_linearization == "numeric_central" :
            #     A,B = self.diff_numeric_central(x,u)
            # elif self.type_linearization == "numeric_forward" :
            #     A,B = self.diff_numeric(x,u)
            # elif self.type_linearization == "analytic" :
            A,B = self.diff(x,u)
            A,B = tf*A,tf*B
            dpdt = np.matmul(A,Phi).reshape((length,ix*ix)).transpose()
            dbmdt = (A@x3 + B*alpha).reshape((length,ix*iu)).transpose()
            dbpdt = (A@x4 + B*beta).reshape((length,ix*iu)).transpose()
            dsdt = np.squeeze(A@x5 + np.expand_dims(f,2)).transpose()
            # dzdt = np.squeeze(A@x6 - A@np.expand_dims(x,2) - B@np.expand_dims(u,2)).transpose()
            # dv = np.vstack((tf*f.transpose(),dpdt,dbmdt,dbpdt,dsdt,dzdt))
            dv = np.vstack((tf*f.transpose(),dpdt,dbmdt,dbpdt,dsdt))
            return dv.flatten(order='F')
        
        A0 = np.eye(ix).flatten()
        Bm0 = np.zeros((ix*iu))
        Bp0 = np.zeros((ix*iu))
        s0 = np.zeros(ix)
        # z0 = np.zeros(ix)
        # V0 = np.array([np.hstack((x[i],A0,Bm0,Bp0,s0,z0)) for i in range(N)]).transpose()
        V0 = np.array([np.hstack((x[i],A0,Bm0,Bp0,s0)) for i in range(N)]).transpose()
        V0_repeat = V0.flatten(order='F')

        # sol = solve_ivp(dvdt,(0,dtau),V0_repeat,args=(u[0:N],u[1:],N),rtol=1e-6,atol=1e-10)
        # sol = solve_ivp(dvdt,(0,dtau),V0_repeat,args=(u[0:N],u[1:],N))
        sol = RK4(dvdt,(0,dtau),V0_repeat,args=(u[0:N],u[1:],N),N_RK=50)

        sol = sol.y[:,-1].reshape((N,-1))
        x_prop = sol[:,idx_state].reshape((-1,ix))
        A = sol[:,idx_A].reshape((-1,ix,ix))
        Bm = sol[:,idx_Bm].reshape((-1,ix,iu))
        Bp = sol[:,idx_Bp].reshape((-1,ix,iu))
        s = sol[:,idx_s].reshape((-1,ix,1)).squeeze()
        # z = sol[:,idx_z].reshape((-1,ix,1)).squeeze()
        z = x_prop - np.squeeze(A@np.expand_dims(x[0:N,:],2) +
                            Bm@np.expand_dims(u[0:N,:],2) + 
                            Bp@np.expand_dims(u[1:N+1,:],2) + 
                            np.expand_dims(tf*s,2))

        return A,Bm,Bp,s,z,x_prop