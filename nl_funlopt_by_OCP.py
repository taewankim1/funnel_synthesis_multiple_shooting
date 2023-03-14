from scipy.integrate import solve_ivp
import numpy as np
import cvxpy as cvx
from cvxpy import vec
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))

class nl_funlopt_by_OCP :
    def __init__(self,ix,iu,iq,ip,iw,N,delT,myLMI,myScaling,myModel,max_iter=5,
        w_tr=0) :
        self.ix = ix
        self.iu = iu
        self.iq = iq
        self.ip = ip
        self.iw = iw
        self.delT = delT
        self.N = N
        self.small = 1e-8
        self.w_tr = w_tr

        self.myModel = myModel
        self.myLMI = myLMI
        self.Sx,self.iSx,self.sx,self.Su,self.iSu,self.su = myScaling.get_scaling()
        self.max_iter = max_iter


    def cvx_initialize(self,alpha,lambda_mu,Qini=None,Qf=None,
        Qmax=None,Rmax=None,
        const_state=None,const_input=None) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw
        delT = self.delT

        # optimization variables
        Qcvx = []
        Ycvx = []
        Z11cvx = []
        for i in range(N+1) :
            Qcvx.append(cvx.Variable((ix,ix), PSD=True))
            Ycvx.append(cvx.Variable((iu,ix)))
            Z11cvx.append(cvx.Variable((ix,ix)))
        nu_p = cvx.Variable(N+1,pos=True)
        nu_Q = cvx.Variable(N+1)
        nu_K = cvx.Variable(N+1)
        sv = cvx.Variable(N+1,pos=True) # support value

        # parameters
        Aq,Bm,Bp,Sm,Sp = [],[],[],[],[]
        for i in range(N) :
            Aq.append(cvx.Parameter((ix*ix,ix*ix)))
            Bm.append(cvx.Parameter((ix*ix,iu*ix)))
            Bp.append(cvx.Parameter((ix*ix,iu*ix)))
            Sm.append(cvx.Parameter((ix*ix,ix*ix)))
            Sp.append(cvx.Parameter((ix*ix,ix*ix)))

        # parameters
        F = []
        for i in range(N+1) :
            F.append(cvx.Parameter((ix,iw)))
        C = cvx.Parameter((iq,ix))
        D = cvx.Parameter((iq,iu))
        E = cvx.Parameter((ix,ip))
        G = cvx.Parameter((iq,iw))

        gamma_inv_squared = []
        for i in range(N+1) :
            gamma_inv_squared.append(cvx.Parameter(pos=True))

        constraints = []
        # PSD on Z - LMI
        def stack_LMI(LMI11,LMI21,LMI31,LMI41,LMI22,LMI32,LMI42,LMI33,LMI43,LMI44) :
            row1 = cvx.hstack((LMI11,LMI21.T,LMI31.T,LMI41.T))
            row2 = cvx.hstack((LMI21,LMI22,LMI32.T,LMI42.T))
            row3 = cvx.hstack((LMI31,LMI32,LMI33,LMI43.T))
            row4 = cvx.hstack((LMI41,LMI42,LMI43,LMI44))
            LMI = cvx.vstack((row1,row2,row3,row4))
            return LMI

        for i in range(N+1) :
            Qi = self.Sx@Qcvx[i]@self.Sx # i
            Yi = self.Su@Ycvx[i]@self.Sx
            Ni = C@Qi + D@Yi

            LMI11 = self.Sx@Z11cvx[i]@self.Sx
            LMI21 = - nu_p[i] * E.T
            LMI31 = - F[i].T
            LMI41 = - Ni
            LMI22 = nu_p[i] * np.eye(ip)
            LMI32 = np.zeros((iw,ip))
            LMI42 = np.zeros((iq,ip))
            LMI33 = lambda_mu * np.eye(iw)
            LMI43 = -G
            LMI44 = nu_p[i] * gamma_inv_squared[i] * np.eye(iq)
            LMI_1 = stack_LMI(LMI11,LMI21,LMI31,LMI41,LMI22,LMI32,LMI42,LMI33,LMI43,LMI44)
            constraints.append(LMI_1 >> 0)

        # Linear matrix equality
        for i in range(N) :
            Qi = self.Sx@Qcvx[i]@self.Sx # Q_i
            Yi = self.Su@Ycvx[i]@self.Sx
            Z11i = self.Sx@Z11cvx[i]@self.Sx
            Qip = self.Sx@Qcvx[i+1]@self.Sx # Q_i+1
            Yip = self.Su@Ycvx[i+1]@self.Sx
            Z11ip = self.Sx@Z11cvx[i+1]@self.Sx

            constraints.append(vec(Qip) == Aq[i]@vec(Qi)
                + Bm[i]@vec(Yi) + Bp[i]@vec(Yip)
                + Sm[i]@vec(Z11i) + Sp[i]@vec(Z11ip)
                )

        # constraints on sv
        for i in range(N+1) :
            constraints.append(sv[i] <= 1)
            if i > 0 :
                constraints.append(sv[i] * np.exp(-alpha*delT*i) <= sv[0])

        # constraints on Q
        for i in range(N+1) :
            Qi = self.Sx@Qcvx[i]@self.Sx # Q_i
            constraints.append(Qi >> np.eye(ix)*self.small) # PD
            constraints.append(Qi << nu_Q[i]*np.eye(ix))
            if Qmax is not None :
                constraints.append(Qi << sv[i]*Qmax[i])

        if const_state is not None :
            for const in const_state :
                for i in range(N+1) :
                    Qi = self.Sx@Qcvx[i]@self.Sx 
                    tmp = (sv[i] * const['(b-ax)^2'][i])[np.newaxis,np.newaxis]
                    tmp1 = cvx.hstack((tmp,const['a'][i].T@Qi))
                    tmp2 = cvx.hstack((Qi.T@const['a'][i],Qi))
                    constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)
            

        # constraints on Y
        if const_input is not None :
            for const in const_input :
                for i in range(N+1) :
                    Qi = self.Sx@Qcvx[i]@self.Sx # Q_i
                    Yi = self.Su@Ycvx[i]@self.Sx
                    tmp = (sv[i] * const['(b-au)^2'][i])[np.newaxis,np.newaxis]
                    tmp1 = cvx.hstack((tmp,const['a'][i].T@Yi))
                    tmp2 = cvx.hstack((Yi.T@const['a'][i],Qi))
                    constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)
        if Rmax is not None :
            for i in range(N+1) :
                Qi = self.Sx@Qcvx[i]@self.Sx # Q_i
                Yi = self.Su@Ycvx[i]@self.Sx
                tmp1 = cvx.hstack((Qi,Yi.T))
                tmp2 = cvx.hstack((Yi,sv[i]*Rmax[i]))
                constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)
        for i in range(N+1) :
            Yi = self.Su@Ycvx[i]@self.Sx
            Qi = self.Sx@Qcvx[i]@self.Sx
            tmp1 = cvx.hstack((nu_K[i]*np.eye(iu),Yi))
            tmp2 = cvx.hstack((Yi.T,Qi))
            constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)

        # boundary condition
        if Qini is not None :
            Qi = self.Sx@Qcvx[0]@self.Sx    
            constraints.append(Qi >> sv[0]*Qini)
        if Qf is not None :
            Qi = self.Sx@Qcvx[-1]@self.Sx # Q_i
            constraints.append(Qi << sv[-1]*Qf)
        # objective

        Q0 = self.Sx@Qcvx[0]@self.Sx # Q_i
        l = 1e3*sv[0] + 1e-1*(-cvx.log_det(Q0)) + 1e-1*cvx.sum(nu_Q) + 0*cvx.sum(nu_K)

        self.prob = cvx.Problem(cvx.Minimize(l),constraints)
        print("Is DPP? ",self.prob.is_dcp(dpp=True))

        # save variables
        self.cvx_variables = {}
        self.cvx_variables['Qcvx'] = Qcvx
        self.cvx_variables['Ycvx'] = Ycvx
        self.cvx_variables['Zcvx'] = Z11cvx
        self.cvx_variables['nu_p'] = nu_p
        self.cvx_variables['nu_Q'] = nu_Q
        self.cvx_variables['nu_K'] = nu_K
        self.cvx_variables['sv'] = sv

        # save params
        self.cvx_params = {}
        self.cvx_params['Aq'] = Aq
        self.cvx_params['Bm'] = Bm
        self.cvx_params['Bp'] = Bp
        self.cvx_params['Sm'] = Sm
        self.cvx_params['Sp'] = Sp
        self.cvx_params['C'] = C
        self.cvx_params['D'] = D
        self.cvx_params['E'] = E
        self.cvx_params['F'] = F
        self.cvx_params['G'] = G
        self.cvx_params['gamma_inv_squared'] = gamma_inv_squared

        # save cost
        self.cvx_cost = {}
        self.cvx_cost['l'] = l

    def cvxopt(self,gamma,C,D,E,F,G) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw

        for i in range(N) :
            self.cvx_params['Aq'][i].value = self.Aq[i]
            self.cvx_params['Bm'][i].value = self.Bm[i]
            self.cvx_params['Bp'][i].value = self.Bp[i]
            self.cvx_params['Sm'][i].value= self.Sm[i]
            self.cvx_params['Sp'][i].value = self.Sp[i]
        for i in range(N+1) :
            self.cvx_params['F'][i].value = F[i]
            self.cvx_params['gamma_inv_squared'][i].value = 1 / (gamma[i]**2)
        self.cvx_params['C'].value = C
        self.cvx_params['D'].value = D
        self.cvx_params['E'].value = E
        self.cvx_params['G'].value = G

        self.prob.solve(solver=cvx.MOSEK,ignore_dpp=True)
        Qnew = []
        Znew = []
        Ynew = []
        for i in range(N+1) :
            Qnew.append(self.Sx@self.cvx_variables['Qcvx'][i].value@self.Sx)
            Ynew.append(self.Su@self.cvx_variables['Ycvx'][i].value@self.Sx)
            Znew.append(self.Sx@self.cvx_variables['Zcvx'][i].value@self.Sx)
        Knew = []
        for i in range(N+1) :
            Knew.append(Ynew[i]@np.linalg.inv(Qnew[i]))
        Knew = np.array(Knew)
        Qnew = np.array(Qnew)
        Ynew = np.array(Ynew)
        svnew = self.cvx_variables['sv'].value
        return Qnew,Knew,Ynew,Znew,svnew,self.prob.status,self.cvx_cost['l'].value

    def run(self,xnom,unom,Qnom,Ynom,Znom,gamma,C,D,E,F,G) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw
        delT = self.delT

        self.Q = Qnom
        self.Y = Ynom
        self.Z = Znom

        # if flag_FOH is False :
        #     Anom,Bnom = None,None
        # else :
        #     Anom,Bnom = self.myModel.diff(xnom,unom)

        history = []

        # iteration starts
        for iteration in range(self.max_iter) :
            history_iter = {}
            # step 1. differentiate dynamics
            self.Aq,self.Bm,self.Bp,self.Sm,self.Sp,x_prop,_ = self.myLMI.discrete_foh(
                xnom,
                unom,
                self.Q,self.Y,self.Z,
                delT,self.myModel,
                )
            eps_machine = np.finfo(float).eps
            self.Aq[np.abs(self.Aq) < eps_machine] = 0
            self.Bm[np.abs(self.Bm) < eps_machine] = 0
            self.Bp[np.abs(self.Bp) < eps_machine] = 0
            self.Sm[np.abs(self.Sm) < eps_machine] = 0
            self.Sp[np.abs(self.Sp) < eps_machine] = 0

# return Qnew,Knew,Ynew,Znew,svnew,self.prob.status,self.cvx_cost['l'].value

            # step2. cvxopt
            self.Qnew,self.Knew,self.Ynew,self.Znew,self.svnew,status,l = self.cvxopt(gamma,
                C,D,E,F,G) 

            # step3. evaluation
            # write a code!
            self.Q = self.Qnew
            self.K = self.Knew
            self.Y = self.Ynew
            self.Z = self.Znew
            self.sv = self.svnew
            self.c = l

        return self.Q,self.K,self.Y,self.Z,self.sv,self.c




