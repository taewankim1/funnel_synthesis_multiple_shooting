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

def compute_comm_mat(m,n):
    # https://en.wikipedia.org/wiki/Commutation_matrix
    # determine permutation applied by K
    w = np.arange(m*n).reshape((m,n),order='F').T.ravel(order='F')
    # apply this permutation to the rows (i.e. to each column) of identity matrix and return result
    return np.eye(m*n)[w,:]

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

class LMImodel(object) :
    def __init__(self,name,ix,iu,linearization) :
        self.name = name
        self.ix = ix
        self.iu = iu
        self.type_linearization = linearization
        self.comm_mat = compute_comm_mat(iu,ix)

    def forward(self,Q,Y,Z,idx=None):
        print("this is in parent class")
        pass

    def diff(self) :
        print("this is in parent class")
        pass

    def discrete_foh(self,x,u,delT) :
        print("this is in parent class")
        pass

class LMI_linear_systems(LMImodel) :
    def __init__(self,name,ix,iu,alpha,linearization) :
        super().__init__(name,ix,iu,linearization)
        self.alpha = alpha # decay rate
        iq = ix*ix
        self.iq = iq
        iy = ix*iu
        self.iy = iy

        self.idx_x = slice(0,ix)
        self.idx_q = slice(ix,ix+iq)
        self.idx_A = slice(ix+iq,ix+iq+iq*iq)
        self.idx_Bm = slice(ix+iq+iq*iq,ix+iq+iq*iq+iq*iy)
        self.idx_Bp = slice(ix+iq+iq*iq+iq*iy,ix+iq+iq*iq+iq*iy+iq*iy)
        self.idx_Sm = slice(ix+iq+iq*iq+iq*iy+iq*iy,ix+iq+iq*iq+iq*iy+iq*iy+iq*iq)
        self.idx_Sp = slice(ix+iq+iq*iq+iq*iy+iq*iy+iq*iq,ix+iq+iq*iq+iq*iy+iq*iy+iq*iq+iq*iq)

    def forward(self,q,y,z,A,B) :
        # Y = np.random.random((4,2,3))
        # print(Y)
        # y = Y.reshape(*Y.shape[:-2], -1,order='F')
        # print(y)
        # Y_ = Y.reshape(*y.shape[:-1],iu,-1,order='F')
        # print(Y_)
        # print(Y - Y)
        assert np.ndim(A) == 2 # not vectorized
        assert np.ndim(A) == np.ndim(B)
        Q = np.reshape(q,(self.ix,self.ix),order='F')
        Y = np.reshape(y,(self.iu,self.ix),order='F')
        Z = np.reshape(z,(self.ix,self.ix),order='F')
        # print_np(A)
        # print_np(B)
        # print_np(Y)
        dQ = A@Q + Q@A.T + B@Y + Y.T@B.T + self.alpha*Q + Z

        if np.ndim(q) == 1 :
            dQ = dQ.flatten(order='F')
        return dQ

    def diff(self,A,B) :
        assert np.ndim(A) == 2
        I = np.eye(self.ix)
        Aq = np.kron(I,A) + np.kron(A,I) + self.alpha * np.kron(I,I)
        Bq = np.kron(I,B) + np.kron(B,I) @ self.comm_mat
        Sq = np.kron(I,I) 
        return Aq,Bq,Sq

    def discrete_foh(self,x,u,Q,Y,Z,delT,traj_model,Anom=None,Bnom=None) :
        ix = self.ix
        iq = self.iq
        iy = self.iy

        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
            assert False
        else :
            N = np.size(x,axis = 0) - 1
        if Anom is not None :
            assert len(Anom) == N+1

        def dvdt(t,V,um,up,ym,yp,zm,zp,Anm=None,Anp=None,Bnm=None,Bnp=None) :
            alpha = (delT - t) / delT
            beta = t / delT
            u = alpha * um + beta * up
            y = alpha * ym + beta * yp
            z = alpha * zm + beta * zp
            

            x = V[self.idx_x]
            q = V[self.idx_q]
            Phi = V[self.idx_A].reshape((iq,iq),order='F')
            Bm = V[self.idx_Bm].reshape((iq,iy),order='F')
            Bp = V[self.idx_Bp].reshape((iq,iy),order='F')
            Sm = V[self.idx_Sm].reshape((iq,iq),order='F')
            Sp = V[self.idx_Sp].reshape((iq,iq),order='F')

            # traj terms
            f = traj_model.forward(x,u).squeeze()
            if Anm is None :
                A,B = traj_model.diff(x,u)
            else :
                A = alpha * Anm + beta * Anp
                B = alpha * Bnm + beta * Bnp

            # funl terms
            F = self.forward(q,y,z,A,B)
            Aq,Bq,Sq = self.diff(A,B)

            dxdt = f
            dqdt = F
            dAdt = Aq@Phi
            dBmdt = (Aq@Bm + Bq*alpha)
            dBpdt = (Aq@Bp + Bq*beta)
            dSmdt = (Aq@Sm + Sq*alpha)
            dSpdt = (Aq@Sp + Sq*beta)
            dV = np.hstack((dxdt,dqdt,dAdt.flatten('F'),
                dBmdt.flatten('F'),dBpdt.flatten('F'),
                dSmdt.flatten('F'),dSpdt.flatten('F')))
            return dV

        Aq,Bm,Bp,Sm,Sp = [],[],[],[],[]
        x_prop,q_prop = [],[]
        def vec(Q) :
            return Q.flatten('F')
        for i in range(N) :
            V0 = np.zeros(ix+iq+iq*iq+iq*iy+iq*iy+iq*iq+iq*iq)
            V0[self.idx_x] = x[i]
            V0[self.idx_q] = vec(Q[i])
            V0[self.idx_A] = np.eye(iq).flatten('F')

            if Anom is None :
                sol = RK4(dvdt,(0,delT),V0,args=(u[i],u[i+1],
                    vec(Y[i]),vec(Y[i+1]),
                    vec(Z[i]),vec(Z[i+1]),
                    ),N_RK=50)
            else :
                sol = RK4(dvdt,(0,delT),V0,args=(u[i],u[i+1],
                    vec(Y[i]),vec(Y[i+1]),
                    vec(Z[i]),vec(Z[i+1]),
                    Anom[i],Anom[i+1],
                    Bnom[i],Bnom[i+1]),N_RK=50)
                # sol = solve_ivp(dvdt,(0,delT),V0,args=(u[i],u[i+1],
                #     vec(Y[i]),vec(Y[i+1]),
                #     vec(Z[i]),vec(Z[i+1]),
                #     Anom[i],Anom[i+1],
                #     Bnom[i],Bnom[i+1]),method='RK45',rtol=1e-6,atol=1e-10)
            sol = sol.y[:,-1]

            x_prop.append(sol[self.idx_x])
            q_prop.append(sol[self.idx_q])
            Aq.append(sol[self.idx_A].reshape((iq,iq),order='F'))
            Bm.append(sol[self.idx_Bm].reshape((iq,iy),order='F'))
            Bp.append(sol[self.idx_Bp].reshape((iq,iy),order='F'))
            Sm.append(sol[self.idx_Sm].reshape((iq,iq),order='F'))
            Sp.append(sol[self.idx_Sp].reshape((iq,iq),order='F'))

        Aq = np.array(Aq)
        Bm = np.array(Bm)
        Bp = np.array(Bp)
        Sm = np.array(Sm)
        Sp = np.array(Sp)
        x_prop = np.array(x_prop)
        q_prop = np.array(q_prop)

        return Aq,Bm,Bp,Sm,Sp,x_prop,q_prop

class LMI_refine(LMImodel) :
    def __init__(self,name,ix,iu,alpha,linearization) :
        super().__init__(name,ix,iu,linearization)
        self.alpha = alpha # decay rate
        ip = ix*ix
        self.ip = ip
        iy = ix*iu
        self.iy = iy

        self.idx_x = slice(0,ix)
        self.idx_p = slice(ix,ix+ip)
        self.idx_A = slice(ix+ip,ix+ip+ip*ip)
        self.idx_Sm = slice(ix+ip+ip*ip,ix+ip+2*ip*ip)
        self.idx_Sp = slice(ix+ip+2*ip*ip,ix+ip+3*ip*ip)

    def forward(self,p,z,A_cl) :
        assert np.ndim(A_cl) == 2 # not vectorized
        P = np.reshape(p,(self.ix,self.ix),order='F')
        Z = np.reshape(z,(self.ix,self.ix),order='F')
        dP = A_cl.T@P + P@A_cl + self.alpha*P + Z

        if np.ndim(p) == 1 :
            dP = dP.flatten(order='F')
        return dP

    def diff(self,A_cl) :
        assert np.ndim(A_cl) == 2
        I = np.eye(self.ix)
        Aq = np.kron(I,A_cl.T) + np.kron(A_cl.T,I) + self.alpha * np.kron(I,I)
        Sq = np.kron(I,I) 
        return Aq,Sq

    def discrete_foh(self,x,u,P,K,Z,delT,traj_model,Anom=False,Bnom=False) :
        ix = self.ix
        ip = self.ip
        iy = self.iy

        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
            assert False
        else :
            N = np.size(x,axis = 0) - 1
        if Anom is not None :
            assert len(Anom) == N+1

        def dvdt(t,V,um,up,zm,zp,Km,Kp,Anm=None,Anp=None,Bnm=None,Bnp=None) :
            alpha = (delT - t) / delT
            beta = t / delT
            u = alpha * um + beta * up
            z = alpha * zm + beta * zp
            K = alpha * Km + beta * Kp

            

            x = V[self.idx_x]
            p = V[self.idx_p]
            Phi = V[self.idx_A].reshape((ip,ip),order='F')
            Sm = V[self.idx_Sm].reshape((ip,ip),order='F')
            Sp = V[self.idx_Sp].reshape((ip,ip),order='F')

            # traj terms
            f = traj_model.forward(x,u).squeeze()
            if Anm is None :
                A,B = traj_model.diff(x,u)
            else :
                A = alpha * Anm + beta * Anp
                B = alpha * Bnm + beta * Bnp

            # funl terms
            # print_np(B)

            # print_np(K)
            F = self.forward(p,z,A+B@K)
            Ap,Sq = self.diff(A+B@K)

            dxdt = f
            dpdt = F
            dAdt = Ap@Phi
            dSmdt = (Ap@Sm + Sq*alpha)
            dSpdt = (Ap@Sp + Sq*beta)
            dV = np.hstack((dxdt,dpdt,dAdt.flatten('F'),
                dSmdt.flatten('F'),dSpdt.flatten('F')))
            return dV

        Ap,Sm,Sp = [],[],[]
        x_prop,p_prop = [],[]
        def vec(P) :
            return P.flatten('F')
        for i in range(N) :
            V0 = np.zeros(ix+ip+3*ip*ip)
            V0[self.idx_x] = x[i]
            V0[self.idx_p] = vec(P[i])
            V0[self.idx_A] = np.eye(ip).flatten('F')

            if Anom is None :
                sol = RK4(dvdt,(0,delT),V0,args=(u[i],u[i+1],
                    vec(Z[i]),vec(Z[i+1]),
                    K[i],K[i+1],
                    ),N_RK=50)
            else :
                sol = RK4(dvdt,(0,delT),V0,args=(u[i],u[i+1],
                    vec(Z[i]),vec(Z[i+1]),
                    K[i],K[i+1],
                    Anom[i],Anom[i+1],
                    Bnom[i],Bnom[i+1]),N_RK=50)
                # sol = solve_ivp(dvdt,(0,delT),V0,args=(u[i],u[i+1],
                #     vec(Y[i]),vec(Y[i+1]),
                #     vec(Z[i]),vec(Z[i+1]),
                #     Anom[i],Anom[i+1],
                #     Bnom[i],Bnom[i+1]),method='RK45',rtol=1e-6,atol=1e-10)
            sol = sol.y[:,-1]

            x_prop.append(sol[self.idx_x])
            p_prop.append(sol[self.idx_p])
            Ap.append(sol[self.idx_A].reshape((ip,ip),order='F'))
            Sm.append(sol[self.idx_Sm].reshape((ip,ip),order='F'))
            Sp.append(sol[self.idx_Sp].reshape((ip,ip),order='F'))

        Ap = np.array(Ap)
        Sm = np.array(Sm)
        Sp = np.array(Sp)
        x_prop = np.array(x_prop)
        p_prop = np.array(p_prop)

        return Ap,Sm,Sp,x_prop,p_prop

# Aran = np.random.randn(ix,ix)
# Bran = np.random.randn(ix,iu)

# qran = np.random.randn(ix*ix)
# yran = np.random.randn(ix*iu)
# zran = np.random.randn(ix*ix)

# Aq,Bq,Sq = myLMI.diff(Aran,Bran)

# h = h = pow(2,-17) / 2 
# eps_x = np.identity(ix*ix)
# q_aug_m = np.tile(qran,(ix*ix,1)) - eps_x * h
# q_aug_p = np.tile(qran,(ix*ix,1)) + eps_x * h

# f_m = []
# for q in q_aug_m :
#     f_m.append(myLMI.forward(q,yran,zran,Aran,Bran))
# f_m = np.array(f_m).T

# f_p = []
# for q in q_aug_p :
#     f_p.append(myLMI.forward(q,yran,zran,Aran,Bran))
# f_p = np.array(f_p).T

# diff = (f_p - f_m)/(2*h) - Aq

# print(np.sum(np.abs(diff)))

# eps_u = np.identity(iu*ix)
# y_aug_m = np.tile(yran,(iu*ix,1)) - eps_u * h
# y_aug_p = np.tile(yran,(iu*ix,1)) + eps_u * h

# f_m = []
# for y in y_aug_m :
#     f_m.append(myLMI.forward(qran,y,zran,Aran,Bran))
# f_m = np.array(f_m).T

# f_p = []
# for y in y_aug_p :
#     f_p.append(myLMI.forward(qran,y,zran,Aran,Bran))
# f_p = np.array(f_p).T

# diff = (f_p - f_m)/(2*h) - Bq

# print(np.sum(np.abs(diff)))