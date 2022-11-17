#!pip uninstall tensorflow
#!pip install sciann
import numpy as np 
from numpy import pi
from numpy import savetxt
from numpy import asarray
from time import time
import matplotlib 
import matplotlib.pyplot as plt
import sciann as sn 
import pandas as pd

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#Lame parameters lmbd and Mu (used to define force and hence verify output result)
lmbd = 1.0
mu = 0.5
Q = 4.0

#test grid size (100X100)
Nt = 100
x_test, y_test = np.meshgrid(np.linspace(0,1,Nt), np.linspace(0,1,Nt))

#calculation of actual displacements, stresses and strains
u_act = np.cos(2*pi*x_test) * np.sin(pi*y_test)
v_act = np.sin(pi*x_test) * Q * y_test**4/4
Exx_act = -2*pi*np.sin(2*pi*x_test) * np.sin(pi*y_test)
Eyy_act = 4.0*y_test**3*np.sin(pi*x_test)
Exy_act = 0.5*pi*y_test**4*np.cos(pi*x_test) + 0.5*pi*np.cos(2*pi*x_test)*np.cos(pi*y_test)
Sxx_act = 4.0*y_test**3*np.sin(pi*x_test) - 4.0*pi*np.sin(2*pi*x_test)*np.sin(pi*y_test)
Syy_act = 8.0*y_test**3*np.sin(pi*x_test) - 2.0*pi*np.sin(2*pi*x_test)*np.sin(pi*y_test)
Sxy_act = 0.5*pi*y_test**4*np.cos(pi*x_test) + 0.5*pi*np.cos(2*pi*x_test)*np.cos(pi*y_test)

uv_act = np.sqrt(u_act**2 + v_act**2)
def cust_plot_uv_actual(ax, val, label):
        im = ax.pcolor(x_test, y_test, val, cmap='seismic',vmin=np.abs(val).min(), vmax=np.abs(val).max(),shading='auto')
        ax.set_title(label, fontsize=30, weight='bold')
        ax.axes.xaxis.set_ticklabels([]) 
        cb = plt.colorbar(im, ax=ax)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(20)

actf=['sigm','gelu','soft','elu','atan','tanh']
Nl = [17, 33, 65, 129, 257, 513, 1025, 2049]
#Element (grid) width
h = [1/(N-1) for N in Nl]
ht=[0]*len(Nl);

train_t2=[]

for a in actf:
    print('\n',a,'\n')
    test_data = []
    
    #Calculation of errors
    uv_err_l2_norm = len(Nl)*[0]
    S_err_l2_norm = len(Nl)*[0]
    uv_err_energy_norm = len(Nl)*[0]
    abs_err_data = []
    l2_err_data = []
    energy_err_data = []
    en_rel_uv_err_point = np.zeros_like(u_act)
    uv_err_point=np.zeros_like(u_act)
    S_err_point=np.zeros_like(u_act)
    energy_uv_err_point=np.zeros_like(u_act)
    L2d=[]
    L2s=[]
    Ed=[]
    for k,N in enumerate(Nl):
        # Neural Network Setup.
        dtype='float32'
        x = sn.Variable("x", dtype=dtype)
        y = sn.Variable("y", dtype=dtype)

        #activation function list
        #defining neural net architecture (x,y input-->Uxy,Vxy output(displacement vectors in x and y respectively))
        if a=='sigm':
            d='sigmoid'
            Uxy = sn.Functional("Uxy", [x, y], 4*[40], '{}'.format(d))
            Vxy = sn.Functional("Vxy", [x, y], 4*[40], '{}'.format(d)) 
        elif a=='soft':
            d='softplus'
            Uxy = sn.Functional("Uxy", [x, y], 4*[40], '{}'.format(d))
            Vxy = sn.Functional("Vxy", [x, y], 4*[40], '{}'.format(d))
        else:
            Uxy = sn.Functional("Uxy", [x, y], 4*[40], '{}'.format(a))
            Vxy = sn.Functional("Vxy", [x, y], 4*[40], '{}'.format(a))

        #Lame's parameter as parameters of Neural net 
        lmbd = sn.Parameter(np.random.rand(), name='lmbd', inputs=[x,y])
        mu = sn.Parameter(np.random.rand(), name='mu', inputs=[x,y])

        C11 = (2*mu + lmbd)
        C12 = lmbd
        C33 = 2*mu

        #Displacements used to get strains 
        Exx = sn.diff(Uxy, x)
        Eyy = sn.diff(Vxy, y)
        Exy = (sn.diff(Uxy, y) + sn.diff(Vxy, x))*0.5

        #Stresses obtained using Constitutive equations
        Sxx= Exx*C11 + Eyy*C12
        Syy= Eyy*C11 + Exx*C12
        Sxy= Exy*C33

        #forces acting on 2D element
        Fx = - 1.*(4*pi**2*sn.cos(2*pi*x)*sn.sin(pi*y) - Q*y**3*pi*sn.cos(pi*x)) - 0.5*(pi**2*sn.cos(2*pi*x)*sn.sin(pi*y) - Q*y**3*pi*sn.cos(pi*x))- 0.5*8*pi**2*sn.cos(2*pi*x)*sn.sin(pi*y)

        Fy = 1.0*(3*Q*y**2*sn.sin(pi*x) - 2*pi**2*sn.cos(pi*y)*sn.sin(2*pi*x)) - 0.5*(2*pi**2*sn.cos(pi*y)*sn.sin(2*pi*x) + (Q*y**4*pi**2*sn.sin(pi*x))/4) + 0.5*6*Q*y**2*sn.sin(pi*x)

        #Momentum balance equation
        Lx = sn.diff(Sxx, x) + sn.diff(Sxy, y) - Fx
        Ly = sn.diff(Sxy, x) + sn.diff(Syy, y) - Fy

        Du = sn.Data(Uxy)
        Dv = sn.Data(Vxy)
        #making momentum equations eual to zero and Ux and Uy equal to actual Ux and Uy data 
        targets = [Lx, Ly, Du, Dv]

        #forming model
        #loading weights        
        m = sn.SciModel([x, y], targets, load_weights_from='C:\\Users\\minio\\PINN\\Grids\\{}\\weights-{}\\Third run-final\\weights_{}X{}.hdf5'.format(a,a,N-1,N-1))

        test_data.append({
            r'${u_x}$': Uxy.eval([x_test, y_test]),
            r'${u_y}$': Vxy.eval([x_test, y_test]),
            r'$\sigma_{xx}$': Sxx.eval([x_test, y_test]),
            r'$\sigma_{yy}$': Syy.eval([x_test, y_test]),
            r'$\sigma_{xy}$': Sxy.eval([x_test, y_test]),
            r'$\varepsilon_{xx}$': Exx.eval([x_test, y_test]),
            r'$\varepsilon_{yy}$': Eyy.eval([x_test, y_test]),
            r'$\varepsilon_{xy}$': Exy.eval([x_test, y_test])})
        
        
        #calculating absolute error
        u_err   = abs(u_act-test_data[k][r'${u_x}$'])
        v_err   = abs(v_act-test_data[k][r'${u_y}$'])
        Sxx_err = abs(Sxx_act-test_data[k][r'$\sigma_{xx}$'])
        Syy_err = abs(Syy_act-test_data[k][r'$\sigma_{yy}$'])
        Sxy_err = abs(Sxy_act-test_data[k][r'$\sigma_{xy}$'])
        Exx_err = abs(Exx_act-test_data[k][r'$\varepsilon_{xx}$'])
        Eyy_err = abs(Eyy_act-test_data[k][r'$\varepsilon_{yy}$'])
        Exy_err = abs(Exy_act-test_data[k][r'$\varepsilon_{xy}$'])

        uv_err_area = 0
        S_err_area = 0
        energy_uv_err_area = 0

        #L2 norm calculations
        #L2 norm error in displacements
        for i in range(Nt):
            for j in range(Nt):
                #corner points
                if (i== 0 and j==0)or(i== 0 and j==(Nt-1))or(i==(Nt-1) and j==0)or(i== (Nt-1) and j==(Nt-1)):
                    uv_err_area += (h[k]**2)*(u_err[i][j]**2+v_err[i][j]**2)/4
                    uv_err_point[i][j] = np.sqrt((u_err[i][j]**2+v_err[i][j]**2))
                #interior points    
                elif (i!= 0 and i!=Nt-1 and j!=0 and j!=Nt-1): 
                    uv_err_area += (h[k]**2)*(u_err[i][j]**2+v_err[i][j]**2)
                    uv_err_point[i][j] = np.sqrt((u_err[i][j]**2+v_err[i][j]**2))

                #edge points    
                else: 
                    uv_err_area += (h[k]**2)*(u_err[i][j]**2+v_err[i][j]**2)/2
                    uv_err_point[i][j] = np.sqrt((u_err[i][j]**2+v_err[i][j]**2))

        uv_err_l2_norm[k] = np.sqrt(uv_err_area)           
        print("L2 norm error in displacement vectors with {}X{} grid-{} =".format(N-1,N-1,a), uv_err_l2_norm[k])

        #L2 norm error in stresses
        for i in range(Nt):
            for j in range(Nt):
                #corner points
                if (i== 0 and j==0)or(i== 0 and j==(Nt-1))or(i==(Nt-1) and j==0)or(i== (Nt-1) and j==(Nt-1)):
                    S_err_area += (h[k]**2)*(Sxx_err[i][j]**2 + Syy_err[i][j]**2 + Sxy_err[i][j]**2)/4
                    S_err_point[i][j] = np.sqrt(Sxx_err[i][j]**2 + Syy_err[i][j]**2 + Sxy_err[i][j]**2)
                #interior points    
                elif (i!= 0 and i!=Nt-1 and j!=0 and j!=Nt-1): 
                    S_err_area += (h[k]**2)*(Sxx_err[i][j]**2 + Syy_err[i][j]**2 + Sxy_err[i][j]**2)
                    S_err_point[i][j] = np.sqrt(Sxx_err[i][j]**2 + Syy_err[i][j]**2 + Sxy_err[i][j]**2)

                #edge points    
                else: 
                    S_err_area += (h[k]**2)*(Sxx_err[i][j]**2 + Syy_err[i][j]**2 + Sxy_err[i][j]**2)/2
                    S_err_point[i][j] = np.sqrt(Sxx_err[i][j]**2 + Syy_err[i][j]**2 + Sxy_err[i][j]**2)

        S_err_l2_norm[k] = np.sqrt(S_err_area)           
        print("L2 norm error in Stresses with {}X{} grid-{} =".format(N-1,N-1,a), S_err_l2_norm[k])    

        #Error with energy norm calculations
        #Error with energy norm in displacements
        for i in range(Nt):
            for j in range(Nt):
                #corner points
                if (i== 0 and j==0)or(i== 0 and j==(Nt-1))or(i==(Nt-1) and j==0)or(i== (Nt-1) and j==(Nt-1)):
                    energy_uv_err_area += (h[k]**2)*(Exx_err[i][j]*Sxx_err[i][j]+Eyy_err[i][j]*Syy_err[i][j]+(2*Exy_err[i][j]*Sxy_err[i][j]))/4
                    energy_uv_err_point[i][j] = np.sqrt(Exx_err[i][j]*Sxx_err[i][j]+Eyy_err[i][j]*Syy_err[i][j]+(2*Exy_err[i][j]*Sxy_err[i][j]))
                    denom_en = np.sqrt(np.abs(Exx_act[i][j])*np.abs(Sxx_act[i][j])+np.abs(Eyy_act[i][j])*np.abs(Syy_act[i][j])+2*np.abs(Exy_act[i][j])*np.abs(Sxy_act[i][j]))
                    en_rel_uv_err_point[i][j] = energy_uv_err_point[i][j]/denom_en if denom_en != 0 else 0

                #interior points    
                elif (i!= 0 and i!=Nt-1 and j!=0 and j!=Nt-1): 
                    energy_uv_err_area += (h[k]**2)*(Exx_err[i][j]*Sxx_err[i][j]+Eyy_err[i][j]*Syy_err[i][j]+(2*Exy_err[i][j]*Sxy_err[i][j]))
                    energy_uv_err_point[i][j] = np.sqrt(Exx_err[i][j]*Sxx_err[i][j]+Eyy_err[i][j]*Syy_err[i][j]+(2*Exy_err[i][j]*Sxy_err[i][j]))
                    denom_en = np.sqrt(np.abs(Exx_act[i][j])*np.abs(Sxx_act[i][j])+np.abs(Eyy_act[i][j])*np.abs(Syy_act[i][j])+2*np.abs(Exy_act[i][j])*np.abs(Sxy_act[i][j]))
                    en_rel_uv_err_point[i][j] = energy_uv_err_point[i][j]/denom_en if denom_en != 0 else 0
                #edge points    
                else: 
                    energy_uv_err_area += (h[k]**2)*(Exx_err[i][j]*Sxx_err[i][j]+Eyy_err[i][j]*Syy_err[i][j]+(2*Exy_err[i][j]*Sxy_err[i][j]))/2
                    energy_uv_err_point[i][j] = np.sqrt(Exx_err[i][j]*Sxx_err[i][j]+Eyy_err[i][j]*Syy_err[i][j]+(2*Exy_err[i][j]*Sxy_err[i][j]))
                    denom_en = np.sqrt(np.abs(Exx_act[i][j])*np.abs(Sxx_act[i][j])+np.abs(Eyy_act[i][j])*np.abs(Syy_act[i][j])+2*np.abs(Exy_act[i][j])*np.abs(Sxy_act[i][j]))
                    en_rel_uv_err_point[i][j] = energy_uv_err_point[i][j]/denom_en if denom_en != 0 else 0

        uv_err_energy_norm[k] = np.sqrt(energy_uv_err_area)           
        print("Energy norm error in displacement vectors with {}X{} grid-{} =".format(N-1,N-1,a), uv_err_energy_norm[k])
        print('\n')
        #creating errors list
        abs_err_data.append({
            r'$|\Delta{u_x}|$':u_err,
            r'$|\Delta{u_y}|$': v_err,
            r'$|\Delta\sigma_{xx}|$': Sxx_err,
            r'$|\Delta\sigma_{yy}|$': Syy_err,
            r'$|\Delta\sigma_{xy}|$': Sxy_err,
            r'$|\Delta\varepsilon_{xx}|$': Exx_err,
            r'$|\Delta\varepsilon_{yy}|$': Eyy_err,
            r'$|\Delta\varepsilon_{xy}|$': Exy_err
        })
        l2_err_data.append({
            r'$||err(|u|)||_{L^2(\mathcal{B})}$':uv_err_point,
            r'$||err(\sigma)||_{L^2(\mathcal{B})}$': S_err_point,
        })
        energy_err_data.append({
            r'$||err(|u|)||_{A(\mathcal{B})}$':energy_uv_err_point, 
            r'$||e(|u|)||_{A}$ (relative)':en_rel_uv_err_point
        })
        L2d.append(np.array(uv_err_l2_norm[k]))
        L2s.append(np.array(S_err_l2_norm[k]))
        Ed.append(np.array(uv_err_energy_norm[k]))
        
    df = pd.DataFrame({"h" : h, "L2 normed error(displacement)" : L2d})
    df.to_csv("hvsL2d-{}.csv".format(a), index=False)
    
    df = pd.DataFrame({"h" : h, "L2 normed error(stress)" : L2s})
    df.to_csv("hvsL2s-{}.csv".format(a), index=False)
    
    df = pd.DataFrame({"h" : h, "Energy normed error(displacement)" : Ed})
    df.to_csv("hvsEd-{}.csv".format(a), index=False)

#convergence ratio cakculations
v_conv_ratio = []
for index,N in enumerate(Nl):
    v_act_max = np.amax(abs((v_act)))
    v_act_i = np.argmax(abs((v_act)))
    if (v_act_i+1)%Nt!=0:
        v_act_max_row = int((v_act_i+1)/Nt)
        v_act_max_col = ((v_act_i+1)%Nt)-1
    else:
        v_act_max_row = int((v_act_i+1)/Nt)-1
        v_act_max_col = Nt-1
    v_test_max = test_data[index][r'${u_y}$'][v_act_max_row][v_act_max_col]
    v_conv_ratio.append(np.abs(v_test_max)/v_act_max)
print("Convergence ratio in uv: {}".format(v_conv_ratio))
print(type(v_conv_ratio))
df = pd.DataFrame({"h" : h, "convergence ratio in y (displacement)" : v_conv_ratio})
df.to_csv("hvsCry-{}.csv".format(a), index=False)
print(v_act_max_row,v_act_max_col)

actual_data = {
    r'${u_x^\star}$':u_act,
    r'${u_y^\star}$': v_act,
    r'$\sigma_{xx}^\star$': Sxx_act,
    r'$\sigma_{yy}^\star$': Syy_act,
    r'$\sigma_{xy}^\star$': Sxy_act,
    r'$\varepsilon_{xx}^\star$': Exx_act,
    r'$\varepsilon_{yy}^\star$': Eyy_act,
    r'$\varepsilon_{xy}^\star$': Exy_act    
}

#contour plots for net displacement
for index,N in enumerate(Nl):
    def cust_plot_uv_actual(ax, val, label):
        im = ax.pcolor(x_test, y_test, val, cmap='seismic',
                      vmin=np.abs(val).min(), vmax=np.abs(val).max(),shading='auto')
        ax.set_title(label, fontsize=30, weight='bold')
        ax.axes.xaxis.set_ticklabels([]) 
        cb = plt.colorbar(im, ax=ax)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(20)

    fig,ax = plt.subplots(figsize=(5,5))
    cust_plot_uv_actual(ax, uv_act, r'$|u|^\star$')

    plt.tight_layout()
    plt.savefig('net_disp_actual_output.png'.format(N-1,N-1),dpi=350)

#contour plots for pinn solution
for index,N in enumerate(Nl):
    def cust_plot_test(ax, val, label,N):
        im = ax.pcolor(x_test, y_test, val, cmap='seismic', 
                       vmin=-np.abs(val).max(), vmax=np.abs(val).max(),shading='auto')
        ax.set_title(label, fontsize=30, weight='bold')
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        cb = plt.colorbar(im, ax=ax)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(20)   
    fig, (ax1,ax2) = plt.subplots(2,4,figsize=(20,8), sharey=True)
    j=0
    for i,(key, val) in enumerate(test_data[index].items()):
        if i<4:
            cust_plot_test(ax1[i], val, key,N)
        else:
            cust_plot_test(ax2[j], val, key,N)
            j=j+1
    plt.tight_layout()
    plt.savefig('test output {}X{} grid.png'.format(N-1,N-1),dpi=350)

#contour plots for absolute error
for index,N in enumerate(Nl):
    def cust_plot_error(ax, val, label,N):
        im = ax.pcolor(x_test, y_test, val, cmap='seismic', 
                      vmin=(val).min(), vmax=(val).max(),shading='auto')
        ax.set_title(label, fontsize=30, weight='bold')
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        cb = plt.colorbar(im, ax=ax)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(20)    
    
    fig, (ax1,ax2) = plt.subplots(2,4,figsize=(20,8), sharey=True)
    j=0
    for i, (key, val) in enumerate(abs_err_data[index].items()):
        if i<4:
            cust_plot_error(ax1[i], val, key,N)
        else:
            cust_plot_error(ax2[j], val, key,N)
            j=j+1
    #fig.suptitle(r"(${} \times {}$)".format(N-1,N-1), fontsize=16)
    plt.tight_layout()
    plt.savefig('abs error {}X{} grid.png'.format(N-1,N-1),dpi=350)

#contour plots for l2-norm error
for index,N in enumerate(Nl):
   def cust_plot_error(ax, val, label,N):
       im = ax.pcolor(x_test, y_test, val, cmap='seismic', 
                     vmin= (val).min(), vmax=(val).max(),shading='auto')
       ax.set_title(label, fontsize=30,weight='bold') 
       ax.axes.xaxis.set_ticklabels([])
       ax.axes.yaxis.set_ticklabels([])
       cb = plt.colorbar(im, ax=ax)
       for t in cb.ax.get_yticklabels():
           t.set_fontsize(20)    
   fig, ax = plt.subplots(1,2,figsize=(10,5))
   for i, (key, val) in enumerate(l2_err_data[index].items()):
       cust_plot_error(ax[i], val, key,N)

   plt.tight_layout()
   plt.savefig('l2 error {}X{} grid.png'.format(N-1,N-1),dpi=350)

#contour plots for energy norm error
for index,N in enumerate(Nl):
    def cust_plot_error(ax, val, label,N):
        im = ax.pcolor(x_test, y_test, val, cmap='seismic', 
                      vmin=(val).min(), vmax=(val).max(),shading='auto')
        ax.set_title(label, fontsize=20)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        cb = plt.colorbar(im, ax=ax)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(20)   
    fig, ax = plt.subplots(figsize=(5,5))
    for i, (key, val) in enumerate(energy_err_data[index].items()):
        if i==0:
            cust_plot_error(ax, val, key, N)

    plt.tight_layout()
    plt.savefig('energy error {}X{} grid.png'.format(N-1,N-1))



