import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
#     print ("Values are: \n%s" % (x))

from model import UnicycleModel
from cost import UnicycleCost
from constraints import UnicycleConstraints
from matplotlib.patches import Ellipse
from utils.utils_plot import plot_two_funnel

from Scaling import TrajectoryScaling

from nl_funlopt_by_OCP import nl_funlopt_by_OCP
from model.LMImodel import LMI_linear_systems

# parameter setting
N = 30
tf = 5
delT = tf/N

# System dynamics
myModel = UnicycleModel.unicycle1('unicycle','analytic')

ix = myModel.ix
iu = myModel.iu
iw = myModel.iw
iq = myModel.iq
ip = myModel.ip

C = myModel.C
D = myModel.D
E = myModel.E
G = myModel.G

# cost function
myCost = UnicycleCost.unicycle('Hello',ix,iu,N)

# constraint
myConst = UnicycleConstraints.UnicycleConstraints('Hello',ix,iu)

# boundary condition of nominal trajectory
xi = np.zeros(3)
xi[0] = 0.0
xi[1] = 0.0 
xi[2] = 0

xf = np.zeros(3)
xf[0] = 5.0
xf[1] = 5.0
xf[2] = 0

# load the nominal trajectorty
with open('tnom.npy','rb') as f:
    tnom = np.load(f)
with open('xnom.npy','rb') as f:
    xnom = np.load(f)
with open('unom.npy','rb') as f:
    unom = np.load(f)

# obstacle environment
def get_H_obs(rx,ry) :
    return np.diag([1/rx,1/ry])
c_list = []
H_list = []
c1 = [1.5,2.75]
H1 = get_H_obs(0.5,0.7)
c_list.append(c1)
H_list.append(H1)
c2 = [4,3.5]
H2 = get_H_obs(0.5,0.7)
c_list.append(c2)
H_list.append(H2)

# get A,B,F from nominal trajectory - we already have C,D,E,G
A,B = myModel.diff(xnom,unom)
F = myModel.diff_F(xnom,unom,np.zeros((N+1,iw)))

# Qmax,Rmax - arbitrary large enough
Qmax = []
for i in range(N+1) :
    Qmax.append(np.diag([0.4**2,0.4**2,np.deg2rad(20)**2])*5  ) 
Rmax = myConst.get_Rmax(unom)

# boundary condition on funnel
Qini = np.diag([0.08,0.08,0.06])
Qf = np.diag([0.08,0.08,0.06])

# nominal K, Y for Lipschitz estimation
Ynom,Knom = myConst.get_YKnom(A,B,Qmax,Rmax)

# alpha and lambda
alpha = 0.7
lambda_mu = 0.5

# a,b : first-order approximation of constraints
const_input = myConst.get_const_input(xnom,unom)
const_state = myConst.get_const_state(xnom,unom,c_list,H_list)

# Lipschitz
from Lipschitz import Lipschitz
lip_estimator = Lipschitz(ix,iu,iq,ip,iw,N,num_sample=100,flag_uniform=True)
lip_estimator.initialize(xnom,unom,Qmax,Knom,A,B,C,D,E,F,G,myModel)
gamma = lip_estimator.update_lipschitz_norm(myModel)

# scaling for numerical optimization
x_max = np.array([1,1,np.pi])
x_min = np.zeros(ix)
u_max = np.array([1,1]) 
u_min = np.zeros(iu)
funl_scaling = TrajectoryScaling(x_min,x_max,u_min,u_max,tf)

# declare LMI
myLMI = LMI_linear_systems('test',ix,iu,alpha+lambda_mu,"analytic")

# funnel optimizer
funl_solver1 = nl_funlopt_by_OCP(ix,iu,iq,ip,iw,N,delT,
                             myLMI,funl_scaling,
                             myModel,
                             max_iter=1)
funl_solver1.cvx_initialize(alpha,lambda_mu,Qini=Qini,Qf=Qf,const_state=const_state,const_input=const_input)
Q1,K1,Y1,Z1,sv1,cost1 = funl_solver1.run(xnom,unom,Qmax,Ynom,Qmax,gamma,C,D,E,F,G)
print("cost",cost1)
S1 = [Q/sv for Q,sv in zip(Q1,sv1)]

# propagation to obtain the funnel between node points
from scipy.integrate import solve_ivp
from scipy import interpolate
Yfoh = interpolate.interp1d(tnom,Y1,axis=0)
Zfoh = interpolate.interp1d(tnom,Z1,axis=0)
ufoh = interpolate.interp1d(tnom,unom,axis=0)
svfoh_inv = interpolate.interp1d(tnom,1/sv1,axis=0)
idx_x = slice(0,ix)
idx_q = slice(ix,ix+ix*ix)
def vec(Q) :
    return Q.flatten('F')
def dfdt(t,V) :
    u = ufoh(t)
    x = V[idx_x]
    dxdt = myModel.forward(x,u).squeeze()
    return dxdt
def dFdt(t,V) :
    u = ufoh(t)
    y = vec(Yfoh(t))
    z = vec(Zfoh(t))
    x = V[idx_x]
    q = V[idx_q]
    A,B = myModel.diff(x,u)
    dxdt = myModel.forward(x,u).squeeze()
    dqdt = myLMI.forward(q,y,z,A,B)
    dV = np.hstack((dxdt,dqdt))
    return dV

tfwd = np.linspace(0,tf,1000)
xfwd = []
qfwd = []
for i in range(N) :
    V0 = np.zeros(ix+ix*ix)
    V0[idx_x] = xnom[i]
    V0[idx_q] = vec(Q1[i])
    if i == N - 1 :
        t_eval = tfwd[np.logical_and(tfwd >= delT*i,tfwd <= delT*(i+1))]
    else :
        t_eval = tfwd[np.logical_and(tfwd >= delT*i,tfwd < delT*(i+1))]
    sol = solve_ivp(dFdt,(delT*i,delT*(i+1)),V0,t_eval=t_eval)
    xfwd.append(sol.y.T[:,idx_x])
    qfwd.append(sol.y.T[:,idx_q])
    
xfwd = np.vstack(xfwd)
qfwd = np.vstack(qfwd)
Qfwd = []
for q in qfwd :
    Qfwd.append(q.reshape((ix,ix),order='F'))
Qfwd = np.array(Qfwd)    
ufwd = ufoh(tfwd)
Qfwd_inv = np.linalg.inv(Qfwd)
Yfwd = Yfoh(tfwd)
Kfwd = Yfwd@Qfwd_inv
svfwd = 1/(svfoh_inv(tfwd))  

def get_state_margin_by_funnel(Q,sv) :
    N = len(Q) - 1
    a1 = np.array([1,0,0])
    a2 = np.array([0,1,0])
    a3 = np.array([0,0,1])
    xfunl1 = []
    yfunl1 = []
    tfunl1 = []
    for i in range(N+1) :
        xfunl1.append(np.sqrt(a1.T@Q[i]/sv[i]@a1))
        yfunl1.append(np.sqrt(a2.T@Q[i]/sv[i]@a2))
        tfunl1.append(np.sqrt(a3.T@Q[i]/sv[i]@a3))
    xfunl1 = np.array(xfunl1)
    yfunl1 = np.array(yfunl1)
    tfunl1 = np.array(tfunl1)
    return xfunl1,yfunl1,tfunl1
def get_input_margin_by_funnel(Q,K,sv) :
    N = len(Q) - 1
    R1 = []
    for i in range(N+1) :
        R1.append(K[i]@Q[i]/sv[i]@K[i].T)
    a1 = np.array([1,0])
    a2 = np.array([0,1])
    vfunl1 = []
    wfunl1 = []
    for i in range(N+1) :
        vfunl1.append(np.sqrt(a1.T@R1[i]@a1))
        wfunl1.append(np.sqrt(a2.T@R1[i]@a2))
    vfunl1 = np.array(vfunl1)
    wfunl1 = np.array(wfunl1)
    return vfunl1,wfunl1

xfunl2,yfunl2,thfunl2 = get_state_margin_by_funnel(Qfwd,svfwd)
vfunl2,wfunl2 = get_input_margin_by_funnel(Qfwd,Kfwd,svfwd)

xfunl2_n,yfunl2_n,thfunl2_n = get_state_margin_by_funnel(Q1,sv1)
vfunl2_n,wfunl2_n = get_input_margin_by_funnel(Q1,K1,sv1)
    

plt.figure(figsize=(7,7))
plot_two_funnel(xnom,Q1,S1,xi=xnom[0],xf=xnom[-1],Qi=Qini,Qf=Qf,plt=plt)
plt.grid(True)

ax = plt.gca()
for ce,H in zip(c_list,H_list) :
    rx = 1/H[0,0]
    ry = 1/H[1,1]
    circle1 = Ellipse((ce[0],ce[1]),rx*2,ry*2,color='tab:red',alpha=0.5,fill=True)
    ax.add_patch(circle1)
plt.grid(True)

filepath = ''
plt.savefig(filepath + 'funnel_result.png',bbox_inches='tight')#,pad_inches=0.5)

fS = 15
plt.figure(figsize=(13,3))
plt.subplot(121)
ax = plt.gca()

ax.plot(tfwd, ufwd[:,0]*0+myConst.vmax,'-.',color='tab:red',alpha=1.0,linewidth=2.0,label='limit')
ax.plot(tfwd, ufwd[:,0]*0+myConst.vmin,'-.',color='tab:red',alpha=1.0,linewidth=2.0)
ax.plot(tfwd, ufwd[:,0],'--',color='black',alpha=1.0,linewidth=2.0,label='nominal')

ax.plot(tfwd, ufwd[:,0]+vfunl2,'-',color='tab:blue',alpha=1.0,linewidth=2.0)
ax.plot(tfwd, ufwd[:,0]-vfunl2,'-',color='tab:blue',alpha=1.0,linewidth=2.0)
ax.plot(tnom, unom[:,0]+vfunl2_n,'o',color='tab:blue',alpha=1.0,linewidth=2.0)
ax.plot(tnom, unom[:,0]-vfunl2_n,'o',color='tab:blue',alpha=1.0,linewidth=2.0)

ax.set_xlabel('time (s)', fontsize = fS)
ax.set_ylabel('$u_v$ (m/s)', fontsize = fS)
# ax.set_xticks(fontsize=15)
# ax.set_yticks(fontsize=15)
ax.axis([0.0, tf, -0.5, 2.5])
ax.grid(True)

plt.subplot(122)
ax = plt.gca()
ax.plot(tfwd, ufwd[:,1]*0+myConst.wmax,'-.',color='tab:red',alpha=1.0,linewidth=2.0)
ax.plot(tfwd, ufwd[:,1]*0+myConst.wmin,'-.',color='tab:red',alpha=1.0,linewidth=2.0,label='limit')
ax.plot(tfwd, ufwd[:,1],'--',color='black',alpha=1.0,linewidth=2.0,label='nominal')

ax.plot(tfwd, ufwd[:,1]+wfunl2,'-',color='tab:blue',alpha=1.0,linewidth=2.0,label='funnel')
ax.plot(tnom, unom[:,1]+wfunl2_n,'o',color='tab:blue',alpha=1.0,linewidth=2.0)
ax.plot(tfwd, ufwd[:,1]-wfunl2,'-',color='tab:blue',alpha=1.0,linewidth=2.0)
ax.plot(tnom, unom[:,1]-wfunl2_n,'o',color='tab:blue',alpha=1.0,linewidth=2.0)

ax.set_xlabel('time (s)', fontsize = fS)
ax.set_ylabel('$u_{\Theta}$ (rad/s)', fontsize = fS)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
ax.axis([0.0, tf, -3, 3])
ax.legend(fontsize=fS,loc=1)

ax.grid(True)

filepath = ''
plt.savefig(filepath + 'input_result.png',bbox_inches='tight')#,pad_inches=0.5)

plt.show()