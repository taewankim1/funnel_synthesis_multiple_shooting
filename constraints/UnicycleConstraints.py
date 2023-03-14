import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.linalg
import time
import random
import cvxpy as cvx
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))


from constraints.constraints import OptimalcontrolConstraints

def get_obs_ab(c,H,xbar) :
    hr = 1 - cvx.norm(H@(xbar[0:2]-c))
    dhdr = - (H.T@H@(xbar[0:2]-c)/cvx.norm(H@(xbar[0:2]-c))).T
    a = dhdr
    b = dhdr@xbar[0:2] - hr
    return  a,b

class UnicycleConstraints(OptimalcontrolConstraints):
    def __init__(self,name,ix,iu):
        super().__init__(name,ix,iu)
        self.idx_bc_f = slice(0, ix)
        self.ih = 4

        self.vmax = 2.0
        self.vmin = 0.0

        self.wmax = 2.0
        self.wmin = -2.0


    def set_obstacle(self,c,H) :
        self.c = c
        self.H = H
        
    def forward(self,x,u,xbar,ubar,Q,K,refobs,aQav,aQaw):
        h = []
        # obstacle avoidance
        # def get_obs_const(c1,H1) :
        #     a,b = get_obs_ab(c1,H1,xbar)
        #     h_Q = cvx.sqrt(a.T@Q[0:2,0:2]@a)
        #     return h_Q+a.T@x[0:2] <= b
        # if self.H is not None :
        #     for c1,H1 in zip(self.c,self.H) :
        #         h.append(get_obs_const(c1,H1))

        for obs in refobs :
            h.append(obs[3] + obs[0:2].T@x[0:2]<=obs[2])

        # input constraints
        a = np.expand_dims(np.array([1,0]),1)
        h.append(aQav + a.T@u <= self.vmax)
        h.append(aQav - a.T@u <= -self.vmin)

        a = np.expand_dims(np.array([0,1]),1)
        h.append(aQaw + a.T@u <= self.wmax)
        h.append(aQaw - a.T@u <= -self.wmin)
        return h

    def get_const_state(self,xnom,unom,c_list,H_list) :

        const_state = []
        c,H = c_list[0],H_list[0] # temporary
        M = np.array([[1,0,0],[0,1,0]])
        N = len(xnom) 
        for c,H in zip(c_list,H_list) :
            tmp_zip = {}
            a = np.zeros((N,3,1))
            bb = np.zeros(N)
            for i in range(N) :
                x = xnom[i]
                deriv  = - M.T@H.T@H@(M@x-c) / np.linalg.norm(H@(M@x-c))
                s = 1 - np.linalg.norm(H@(M@x-c))
                a[i,:,0] = deriv
                b = -s + deriv@x
                bb[i] = (b - a[i,:,0].T@x) ** 2
            tmp_zip['a'] = a
            tmp_zip['(b-ax)^2'] = bb
            const_state.append(tmp_zip)
        return const_state


    def get_const_input(self,xnom,unom) :
        N = len(unom)
        const_input = []
        a = np.zeros((N,2,1))
        a[:,0,:] = 1
        b = self.vmax * np.ones(N)
        c = (b - np.squeeze(np.transpose(a,(0,2,1))@unom[:,:,np.newaxis])) ** 2
        tmp_zip = {}
        tmp_zip['a'] = a
        tmp_zip['(b-au)^2'] = c
        const_input.append(tmp_zip)

        a = np.zeros((N,2,1))
        a[:,0,:] = -1
        b = -self.vmin * np.ones(N)
        c = (b - np.squeeze(np.transpose(a,(0,2,1))@unom[:,:,np.newaxis])) ** 2
        tmp_zip = {}
        tmp_zip['a'] = a
        tmp_zip['(b-au)^2'] = c
        const_input.append(tmp_zip)
        
        a = np.zeros((N,2,1))
        a[:,1,:] = 1
        b = self.wmax * np.ones(N)
        c = (b - np.squeeze(np.transpose(a,(0,2,1))@unom[:,:,np.newaxis])) ** 2
        tmp_zip = {}
        tmp_zip['a'] = a
        tmp_zip['(b-au)^2'] = c
        const_input.append(tmp_zip)

        a = np.zeros((N,2,1))
        a[:,1,:] = -1
        b = -self.wmin * np.ones(N)
        c = (b - np.squeeze(np.transpose(a,(0,2,1))@unom[:,:,np.newaxis])) ** 2
        tmp_zip = {}
        tmp_zip['a'] = a
        tmp_zip['(b-au)^2'] = c
        const_input.append(tmp_zip)

        return const_input

    def get_Rmax(self,unom) :
        iu = self.iu
        Rcvx = cvx.Variable((iu,iu),PSD=True)    
        a1 = cvx.Parameter(iu)
        au1 = cvx.Parameter()
        vmin = cvx.Parameter()
        vmax = cvx.Parameter()

        a2 = cvx.Parameter(iu)
        au2 = cvx.Parameter()
        wmin = cvx.Parameter()
        wmax = cvx.Parameter()

        constraint = []
        constraint.append(cvx.norm(Rcvx@a1)+au1 <= vmax)
        constraint.append(cvx.norm(Rcvx@a1)-au1 <= -vmin)

        constraint.append(cvx.norm(Rcvx@a2)+au2 <= wmax)
        constraint.append(cvx.norm(Rcvx@a2)-au2 <= -wmin)
        cost = -cvx.log_det(Rcvx)
        prob = cvx.Problem(cvx.Minimize(cost),constraint)
        # print("Is DPP? ",prob.is_dcp(dpp=True))

        N = len(unom)
        Rmax = []
        for idx in range(N) :
            a1.value = np.array([1,0])
            au1.value = np.array([1,0])@unom[idx]
            vmax.value = self.vmax
            vmin.value = self.vmin

            a2.value = np.array([0,1])
            au2.value = np.array([0,1])@unom[idx]
            wmax.value = self.wmax
            wmin.value = self.wmin

            prob.solve(solver=cvx.MOSEK)
        #     print(idx,prob.status)
            Rmax.append(Rcvx.value@Rcvx.value)
            # Rmax.append(Rcvx.value)
        return Rmax

    def get_YKnom(self,A,B,Qmax,Rmax) :
        ix = self.ix
        iu = self.iu
        N = len(Qmax) - 1
        # TODO - we need to scale the cvx variables when it comes to more complex systems
        Ynom = []
        Knom = []
        for i in range(N+1) :
            constraints = []
            Qi = cvx.Variable((ix,ix),PSD=True)
            Yi = cvx.Variable((iu,ix))
            tmp1 = cvx.hstack((Qi,Yi.T))
            tmp2 = cvx.hstack((Yi,Rmax[i]))
            constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)
            constraints.append( A[i]@Qi+B[i]@Yi + Qi@A[i].T + Yi.T@B[i].T + 0.1 * Qi << 0)
            constraints.append(Qi << Qmax[i])
            prob = cvx.Problem(cvx.Minimize(-cvx.log_det(Qi)),constraints)
            prob.solve(solver=cvx.MOSEK)
            # print(prob.status)
            Ynom.append(Yi.value)
            Knom.append(Yi.value@np.linalg.inv(Qi.value))
            # print(np.linalg.eig(Qi.value))
        return Ynom,Knom




        


