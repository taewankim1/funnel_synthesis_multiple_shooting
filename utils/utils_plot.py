import matplotlib.pyplot as plt
import numpy as np
import time
import random
from matplotlib.patches import Ellipse,Rectangle
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
#     print ("Values are: \n%s" % (x))
from utils.utils_alg import get_radius_angle

# def plot traj
def plot_state_input(t,x,u,xi,xf,N,delT,alpha,plt,flag_step=False) :
    fS = 15
    # plt.figure(idx_plot,figsize=(10,15))
    t_index = np.array(range(N+1))*delT
    plt.subplot(321)
    plt.plot(x[:,0], x[:,1],color='tab:blue',alpha=alpha,linewidth=2.0)
    if xf is not None :
        plt.plot(xf[0],xf[1],"o",label='goal')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis([-0.5, 5.5, -0.5, 5.5])
    plt.xlabel('X (m)', fontsize = fS)
    plt.ylabel('Y (m)', fontsize = fS)
    plt.subplot(322)
    plt.plot(t, x[:,0],color='tab:blue',alpha=alpha, linewidth=2.0,label='naive')
    plt.xlabel('time (s)', fontsize = fS)
    plt.ylabel('x1 (m)', fontsize = fS)
    plt.subplot(323)
    plt.plot(t, x[:,1],color='tab:blue',alpha=alpha, linewidth=2.0,label='naive')
    plt.xlabel('time (s)', fontsize = fS)
    plt.ylabel('x2 (m)', fontsize = fS)
    plt.subplot(324)
    plt.plot(t, x[:,2],color='tab:blue',alpha=alpha, linewidth=2.0,label='naive')
    plt.xlabel('time (s)', fontsize = fS)
    plt.ylabel('x3 (rad)', fontsize = fS)
    plt.subplot(325)
    if flag_step == True :
        plt.step(t_index, [*u[:N,0],u[N-1,0]],color='tab:blue',alpha=alpha,where='post',linewidth=2.0)
    else :
        plt.plot(t, u[:,0],color='tab:blue',alpha=alpha,linewidth=2.0)
    plt.xlabel('time (s)', fontsize = fS)
    plt.ylabel('v (m/s)', fontsize = fS)
    plt.subplot(326)
    if flag_step == True :
        plt.step(t_index, [*u[:N,1],u[N-1,1]],color='tab:blue',alpha=alpha,where='post',linewidth=2.0)
    else :
        plt.plot(t, u[:,1],color='tab:blue',alpha=alpha,linewidth=2.0)
    plt.xlabel('time (s)', fontsize = fS)
    plt.ylabel('w (rad/s)', fontsize = fS)
    # plt.show()

def plot_traj(x,u,c_list,H_list,xf=None,idx_plot=0) :
    fS = 18
    plt.figure(idx_plot,figsize=(7,7))
    ax=plt.gca()
    for ce,H in zip(c_list,H_list) :
        rx = 1/H[0,0]
        ry = 1/H[1,1]
        circle1 = Ellipse((ce[0],ce[1]),rx*2,ry*2,color='tab:red',alpha=0.5,fill=True)
        ax.add_patch(circle1)
    plt.plot(x[:,0], x[:,1],'-', linewidth=2.0)
    if xf is not None :
        plt.plot(xf[0],xf[1],"o",label='goal')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis([-1.0, 6.0, -1.0, 6.0])
    plt.xlabel('X (m)', fontsize = fS)
    plt.ylabel('Y (m)', fontsize = fS)

def plot_traj_set(x,u,c_list,H_list,Q,xi=None,xf=None,Qi=None,Qf=None,plt=plt,flag_label=True,fS=15) :
    radius_list,angle_list = get_radius_angle(Q)

    # plt.figure(idx_plot,figsize=(7,7))
    plt.plot(x[:,0], x[:,1],'--',color='tab:orange',alpha=0.8,linewidth=2.0)
    ax=plt.gca()
    if Qi is not None :
        radius_f,angle_f = get_radius_angle([Qi])
        for radius,angle in zip(radius_f,angle_f) :
            ell = Ellipse((xi[0],xi[1]),radius[0]*2,radius[1]*2,angle=np.rad2deg(angle),
            color='tab:green',alpha=0.5,fill=True)
            ax.add_patch(ell)
    if Qf is not None :
        radius_f,angle_f = get_radius_angle([Qf])
        for radius,angle in zip(radius_f,angle_f) :
            ell = Ellipse((xf[0],xf[1]),radius[0]*2,radius[1]*2,angle=np.rad2deg(angle),
            color='tab:green',alpha=0.5,fill=True)
            ax.add_patch(ell)
    for ce,H in zip(c_list,H_list) :
        rx = 1/H[0,0]
        ry = 1/H[1,1]
        circle1 = Ellipse((ce[0],ce[1]),rx*2,ry*2,color='tab:red',alpha=0.5,fill=True)
        ax.add_patch(circle1)
    for x_,radius,angle in zip(x,radius_list,angle_list) :
        ell = Ellipse((x_[0],x_[1]),radius[0]*2,radius[1]*2,angle=np.rad2deg(angle),color='tab:blue',alpha=0.5,fill=True)
        ax.add_patch(ell)
    # if xf is not None :
    #     plt.plot(xf[0],xf[1],"o",label='goal')
    if flag_label == True :
        plt.plot(1e3,1e3,'--',color='tab:orange',label="nominal")
        plt.plot(1e3,1e3,'o',markersize=15,color='tab:blue',label="funnel") 
        plt.plot(1e3,1e3,'o',markersize=15,color='tab:green',label="initial and final ellipsoid") 
        # plt.plot(1e3,1e3,'o',markersize=15,color='tab:green',label="final") 
        plt.plot(1e3,1e3,'o',markersize=15,alpha=0.5,color='tab:red',label="obstacles") 

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis([-1.0, 6.0, -1.0, 6.0])
    plt.xlabel('$r_x$ (m)', fontsize = fS)
    plt.ylabel('$r_y$ (m)', fontsize = fS)
    if flag_label == True :
        plt.legend(fontsize=fS)
    ticks_font = "Times New Roman"
    for label in ax.get_xticklabels():
        label.set_fontproperties(ticks_font)

    for label in ax.get_yticklabels():
        label.set_fontproperties(ticks_font)

def plot_comparison(x,x_l,c_list,H_list,Q,Q_l,xi=None,xf=None,Qi=None,Qf=None,plt=plt,flag_label=True,fS=15) :

    # plt.figure(idx_plot,figsize=(7,7))
    plt.plot(x[:,0], x[:,1],'--',color='tab:orange',alpha=0.8,linewidth=2.0)
    plt.plot(x_l[:,0], x_l[:,1],'-.',color='tab:brown',alpha=0.8,linewidth=2.0)
    ax=plt.gca()
    if Qi is not None :
        radius_f,angle_f = get_radius_angle([Qi])
        for radius,angle in zip(radius_f,angle_f) :
            ell = Ellipse((xi[0],xi[1]),radius[0]*2,radius[1]*2,angle=np.rad2deg(angle),
            color='tab:green',alpha=0.5,fill=True)
            ax.add_patch(ell)
    if Qf is not None :
        radius_f,angle_f = get_radius_angle([Qf])
        for radius,angle in zip(radius_f,angle_f) :
            ell = Ellipse((xf[0],xf[1]),radius[0]*2,radius[1]*2,angle=np.rad2deg(angle),
            color='tab:green',alpha=0.5,fill=True)
            ax.add_patch(ell)
    for ce,H in zip(c_list,H_list) :
        rx = 1/H[0,0]
        ry = 1/H[1,1]
        circle1 = Ellipse((ce[0],ce[1]),rx*2,ry*2,color='tab:red',alpha=0.5,fill=True)
        ax.add_patch(circle1)
    radius_list,angle_list = get_radius_angle(Q)
    for x_,radius,angle in zip(x,radius_list,angle_list) :
        ell = Ellipse((x_[0],x_[1]),radius[0]*2,radius[1]*2,angle=np.rad2deg(angle),color='tab:blue',alpha=0.5,fill=True)
        ax.add_patch(ell)

    radius_list,angle_list = get_radius_angle(Q_l)
    for x_,radius,angle in zip(x_l,radius_list,angle_list) :
        ell = Ellipse((x_[0],x_[1]),radius[0]*2,radius[1]*2,angle=np.rad2deg(angle),color='tab:brown',alpha=1.0,fill=False)
        ax.add_patch(ell)
    if flag_label == True :
        plt.plot(1e3,1e3,'--',color='tab:orange',label="nominal")
        plt.plot(1e3,1e3,'o',markersize=15,color='tab:blue',label="funnel") 
        plt.plot(1e3,1e3,'o',markersize=15,color='tab:green',label="initial and final") 
        plt.plot(1e3,1e3,'o',markersize=15,alpha=0.5,color='tab:red',label="obstacles") 
        plt.plot(1e3,1e3,'-.',color='tab:brown',label="approx nominal")
        plt.plot(1e3,1e3,'o',markersize=15,fillstyle='none',linewidth=4,color='tab:brown',label="approx funnel") 
        # ell = Ellipse((1e3,1e3),radius[0]*2,radius[1]*2,angle=np.rad2deg(angle),color='tab:brown',alpha=1.0,fill=False,label='linear funnel')
        # ax.add_patch(ell)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis([-1.0, 6.0, -1.0, 6.0])
    plt.xlabel('$r_x$ (m)', fontsize = fS)
    plt.ylabel('$r_y$ (m)', fontsize = fS)
    if flag_label == True :
        plt.legend(fontsize=fS)
    ticks_font = "Times New Roman"
    for label in ax.get_xticklabels():
        label.set_fontproperties(ticks_font)

    for label in ax.get_yticklabels():
        label.set_fontproperties(ticks_font)

def plot_two_funnel(x,Q,S,xi=None,xf=None,Qi=None,Qf=None,plt=plt,flag_label=True,fS=15) :
    radius_Q,angle_Q = get_radius_angle(Q)
    radius_S,angle_S = get_radius_angle(S)

    # plt.figure(idx_plot,figsize=(7,7))
    plt.plot(x[:,0], x[:,1],'--',color='black',alpha=0.8,linewidth=1.0)
    ax=plt.gca()
    # for ce,H in zip(c_list,H_list) :
    #     rx = 1/H[0,0]
    #     ry = 1/H[1,1]
    #     circle1 = Ellipse((ce[0],ce[1]),rx*2,ry*2,color='tab:red',alpha=0.5,fill=True)
    #     ax.add_patch(circle1)
    for x_,radius,angle in zip(x,radius_S,angle_S) :
        ell = Ellipse((x_[0],x_[1]),radius[0]*2,radius[1]*2,angle=np.rad2deg(angle),color='tab:blue',alpha=0.5,fill=True)
        ax.add_patch(ell)
    if Qi is not None :
        radius_f,angle_f = get_radius_angle([Qi])
        for radius,angle in zip(radius_f,angle_f) :
            ell = Ellipse((xi[0],xi[1]),radius[0]*2,radius[1]*2,angle=np.rad2deg(angle),
            color='tab:green',alpha=0.5,fill=True)
            ax.add_patch(ell)
    if Qf is not None :
        radius_f,angle_f = get_radius_angle([Qf])
        for radius,angle in zip(radius_f,angle_f) :
            ell = Ellipse((xf[0],xf[1]),radius[0]*2,radius[1]*2,angle=np.rad2deg(angle),
            color='tab:green',alpha=0.5,fill=True)
            ax.add_patch(ell)
    for x_,radius,angle in zip(x,radius_Q,angle_Q) :
        ell = Ellipse((x_[0],x_[1]),radius[0]*2,radius[1]*2,angle=np.rad2deg(angle),color='tab:brown',alpha=1.0,fill=False,linewidth=2.0)
        ax.add_patch(ell)
    if flag_label == True :
        plt.plot(1e3,1e3,'--',color='black',label="nominal")
        plt.plot(1e3,1e3,'o',markersize=15,color='tab:blue',label="invariant funnel $\mathcal{E}_{c}$") 
        # plt.plot(1e3,1e3,'o',markersize=15,fillstyle='none',linewidth=4,color='tab:brown',label="approx funnel") 
        plt.plot(1e3,1e3,'o',markersize=15,fillstyle='none',linewidth=4,color='tab:brown',label="attractive funnel $\mathcal{E}$") 
        # plt.plot(1e3,1e3,'o',markersize=15,color='tab:green',label="final") 
        plt.plot(1e3,1e3,'o',markersize=15,alpha=1.0,color='tab:green',label="initial and final") 
        plt.plot(1e3,1e3,'o',markersize=15,alpha=0.5,color='tab:red',label="obstacles") 

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis([-1.0, 6.0, -1.0, 6.0])
    plt.xlabel('$r_x$ (m)', fontsize = fS)
    plt.ylabel('$r_y$ (m)', fontsize = fS)
    if flag_label == True :
        plt.legend(fontsize=fS)
    ticks_font = "Times New Roman"
    for label in ax.get_xticklabels():
        label.set_fontproperties(ticks_font)

    for label in ax.get_yticklabels():
        label.set_fontproperties(ticks_font)

# plot sample
# t_index = np.array(range(N+1))*delT
# plt.figure(0,figsize=(10,10))
# plt.subplot(221)
# for x_ in xsam :
#     plt.plot(x_[:,0], x_[:,1], linewidth=2.0)
# plt.plot(xf[0],xf[1],"o",label='goal')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.axis([-0.5, 1.5, -0.5, 1.5])
# plt.xlabel('X (m)', fontsize = fS)
# plt.ylabel('Y (m)', fontsize = fS)
# plt.subplot(222)
# for x_ in xsam :
#     plt.plot(t_index, x_[:,0], linewidth=2.0,label='naive')
# plt.xlabel('time (s)', fontsize = fS)
# plt.ylabel('x1 (m)', fontsize = fS)
# plt.subplot(223)
# for x_ in xsam :
#     plt.plot(t_index, x_[:,1], linewidth=2.0,label='naive')
# plt.xlabel('time (s)', fontsize = fS)
# plt.ylabel('x2 (m)', fontsize = fS)
# plt.subplot(224)
# for x_ in xsam :
#     plt.plot(t_index, np.rad2deg(x_[:,2]), linewidth=2.0,label='naive')
# plt.xlabel('time (s)', fontsize = fS)
# plt.ylabel('x3 (deg)', fontsize = fS)
# #     plt.legend(fontsize=fS)


# plt.figure()
# plt.figure(figsize=(15,5))
# plt.subplot(121)
# for u_ in usam :
#     plt.plot(t_index[:N], u_[:N,0], linewidth=2.0)
# plt.xlabel('time (s)', fontsize = fS)
# plt.ylabel('v (m/s)', fontsize = fS)
# plt.subplot(122)
# for u_ in usam :
#     plt.plot(t_index[:N], u_[:N,1], linewidth=2.0)
# plt.xlabel('time (s)', fontsize = fS)
# plt.ylabel('w (rad/s)', fontsize = fS)
# plt.show()
################################ Plot TEST. Don't erase it
# ellipse_list = []
# angle = []
# for phi in np.linspace(0,2*np.pi,30) :
#     for theta in np.linspace(0,2*np.pi,30) :
#         angle.append([phi,theta])
# angle = np.array(angle)
# phi = angle[:,0]
# theta = angle[:,1]

# c_r = 1    
# circle_x = c_r * np.sin(phi) * np.cos(theta)
# circle_y = c_r * np.sin(phi) * np.sin(theta)
# circle_z = c_r * np.cos(phi)
# circle = np.vstack((circle_x,circle_y,circle_z)) 
# for i in range(N+1) :
#     Q_tmp = Q[i]
#     L = np.linalg.cholesky(np.linalg.inv(Q_tmp))
#     ellipse = np.linalg.inv(L.T)@circle
#     ellipse_list.append(ellipse)

# plt.figure(0,figsize=(10,10))
# plot_traj_set(x,u,c_list,H_list,Q,xi=xi,xf=xf,xi_margin=xi_margin,Qf=Qf,idx_plot=0)
# ax=plt.gca()
# for i in range(N+1) :
#     plt.plot(ellipse_list[i][0,:]+x[i,0],ellipse_list[i][1,:]+x[i,1],alpha=0.5,linewidth=2.0,color='tab:grey')
# plt.plot(x[:,0],x[:,1],color='tab:blue',linestyle='-')
# plt.plot(x[:,0],x[:,1],color='tab:blue')
