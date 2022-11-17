#install sciann by uncommenting next line
#!pip install sciann
import numpy as np 
from numpy import pi
from numpy import savetxt
from numpy import asarray
from time import time
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sciann as sn 
import tensorflow.keras.callbacks as callbacks
import pathlib
from tensorflow.keras.callbacks import CSVLogger

#to set figure properties 
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

#Lame parameters lmbd and mu 
lmbd = 1.0
mu = 0.5
Q = 4.0

#generating various grid sizes (16X16, 32X32 etc.)
#l defines how many grids needed (l=1 means only 16X16)
#Nl[0] is starting of grid set, here strating with 16X16
l = 1;
Nl = [0]*l;

#change this for other grid sizes
Nl[0] = 16;
for i in range(1,l):
    Nl[i] = Nl[i-1] *2;
for i in range(l):
    Nl[i] = Nl[i] + 1;

#set number of epochs high enough   
epochs = 20000000
batch_size = 50
print(Nl)
epch=[]
epch.append(epochs)

#defining element (grid) width
h = [1/(N-1) for N in Nl]

savetxt('N points list {}X{}.csv'.format(Nl[0]-1,Nl[-1]-1), Nl, delimiter=',')
savetxt('h list {}X{}.csv'.format(Nl[0]-1,Nl[-1]-1), h, delimiter=',')
savetxt('Epochs {}X{}.csv'.format(Nl[0]-1,Nl[-1]-1), np.array(epch), delimiter=',')

#data points used for training the model (input x,y)
for N in Nl:
    x_data, y_data = np.meshgrid(np.linspace(0.,1.,N), np.linspace(0., 1., N))
    plt.scatter(x_data, y_data)
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.title("Training data {}X{} grid (input to Neural networks)".format(N-1,N-1))
    # plt.savefig('training data with {}X{} grid using ux,uy model.jpg'.format(N-1,N-1))
    #plt.show()

#displacement data used for training (analytical solution)
u_data = np.cos(2*pi*x_data) * np.sin(pi*y_data)
v_data = np.sin(pi*x_data) * Q * y_data**4/4

#creating zero array of same size as input x_data and y_data
Lx_data = np.zeros_like(x_data)
Ly_data = np.zeros_like(y_data)
c1_data = np.zeros_like(y_data)
c2_data = np.zeros_like(y_data)
c3_data = np.zeros_like(y_data)

#analytical solution equations
#from sympy import *
# #def actualvalues():
# x_act= symbols('x_act')
# y_act = symbols('y_act')

# u_act = cos(2*pi*x_act) * sin(pi*y_act)
# v_act = sin(pi*x_act) * Q * y_act**4/4

# Exx_act= diff(u_act, x_act)
# Eyy_act= diff(v_act, y_act)
# Exy_act = (diff(u_act, y_act) + diff(v_act, x_act))*0.5

# Sxx_act= Exx_act*C11 + Eyy_act*C12
# Syy_act= Eyy_act*C11 + Exx_act*C12
# Sxy_act= Exy_act*C33

#number of test grid data points 
Nt = 100
x_test, y_test = np.meshgrid(np.linspace(0,1,Nt), np.linspace(0,1,Nt))

#saving training instance in ht list
#saving test data in list
#saving training time in tain_t2
ht=[0]*len(Nl);
test_data = []
train_t2=[]

for i,N in enumerate(Nl):
    # Neural Network Setup.
    dtype='float32'
    x = sn.Variable("x", dtype=dtype)
    y = sn.Variable("y", dtype=dtype)

    #activation function 
    act_f= ['tanh']
    #defining neural net architecture (x,y input-->Uxy,Vxy output(displacements in x and y))
    Uxy = sn.Functional("Uxy", [x, y], 4*[40], '{}'.format(act_f[0]))
    Vxy = sn.Functional("Vxy", [x, y], 4*[40], '{}'.format(act_f[0])) 

    #Lame's parameters as parameters of Neural net 
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
    Fx = (- 1.*(4*pi**2*sn.cos(2*pi*x)*sn.sin(pi*y) - Q*y**3*pi*sn.cos(pi*x))- 
                0.5*(pi**2*sn.cos(2*pi*x)*sn.sin(pi*y) - Q*y**3*pi*sn.cos(pi*x))        
                    - 0.5*8*pi**2*sn.cos(2*pi*x)*sn.sin(pi*y))

    Fy = (1.0*(3*Q*y**2*sn.sin(pi*x) - 2*pi**2*sn.cos(pi*y)*sn.sin(2*pi*x))-        
                0.5*(2*pi**2*sn.cos(pi*y)*sn.sin(2*pi*x) + (Q*y**4*pi**2*sn.sin(pi*x))/4)+        
                    0.5*6*Q*y**2*sn.sin(pi*x))

    #Momentum balance equation
    Lx = sn.diff(Sxx, x) + sn.diff(Sxy, y) - Fx
    Ly = sn.diff(Sxy, x) + sn.diff(Syy, y) - Fy

    Du = sn.Data(Uxy)
    Dv = sn.Data(Vxy)
    #making momentum equations equal to zero and Ux and Uy equal to actual Ux and Uy data 
    targets = [Lx, Ly, Du, Dv]

    #defining model 
    #saving log data 
    m = sn.SciModel([x, y], targets)
    csv_logger = CSVLogger('log_{}X{}_tanh.csv'.format(N-1,N-1), append=True, separator=';')
    
    #callbacks
    current_file_path = '/work/ws-tmp/g051309-pinn_work/tanh/'
    checkpoint_filepath = (current_file_path+'weights_{}X{}.hdf5'.format(N-1,N-1))
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        verbose=1,
        monitor='loss',
        mode='auto',
        save_best_only=True
    )
    start = time()

    #training model
    #saving weights
    ht[i] = m.train([x_data, y_data], [Lx_data, Ly_data, u_data, v_data], #syy_data], 
                batch_size = batch_size, epochs= epochs, log_parameters={'parameters':[lmbd, mu], 'freq':1},
                adaptive_weights={'method':'NTK', 'freq': 100}, 
                callbacks=[csv_logger, model_checkpoint_callback])
                #,save_weights={'save_weights_to':".PINN\Weights\{}X{}model".format(N-1,N-1), 'freq': epochs})
    train_t2.append(time()-start)

    #Evaluation of trained model using test data
    test_data.append({
        r'${u_x}$': Uxy.eval([x_test, y_test]),
        r'${u_y}$': Vxy.eval([x_test, y_test]),
        r'$\sigma_{xx}$': Sxx.eval([x_test, y_test]),
        r'$\sigma_{yy}$': Syy.eval([x_test, y_test]),
        r'$\sigma_{xy}$': Sxy.eval([x_test, y_test]),
        r'$\varepsilon_{xx}$': Exx.eval([x_test, y_test]),
        r'$\varepsilon_{yy}$': Eyy.eval([x_test, y_test]),
        r'$\varepsilon_{xy}$': Exy.eval([x_test, y_test])
    })

#saving  training time in sec.
savetxt('Train_time {}X{}.csv'.format(Nl[0]-1,Nl[-1]-1), asarray(train_t2), delimiter=',')   
#saving pinn output
savetxt('Uxy_{}X{}-tanh.csv'.format(Nl[0]-1,Nl[-1]-1), asarray(Uxy.eval([x_test, y_test])), delimiter=',') 
savetxt('Vxy_{}X{}-tanh.csv'.format(Nl[0]-1,Nl[-1]-1), asarray(Vxy.eval([x_test, y_test])), delimiter=',') 
savetxt('Sxx_{}X{}-tanh.csv'.format(Nl[0]-1,Nl[-1]-1), asarray(Sxx.eval([x_test, y_test])), delimiter=',') 
savetxt('Syy_{}X{}-tanh.csv'.format(Nl[0]-1,Nl[-1]-1), asarray(Syy.eval([x_test, y_test])), delimiter=',')
savetxt('Sxy_{}X{}-tanh.csv'.format(Nl[0]-1,Nl[-1]-1), asarray(Sxy.eval([x_test, y_test])), delimiter=',')
savetxt('Exx_{}X{}-tanh.csv'.format(Nl[0]-1,Nl[-1]-1), asarray(Exx.eval([x_test, y_test])), delimiter=',')
savetxt('Eyy_{}X{}-tanh.csv'.format(Nl[0]-1,Nl[-1]-1), asarray(Eyy.eval([x_test, y_test])), delimiter=',')
savetxt('Exy_{}X{}-tanh.csv'.format(Nl[0]-1,Nl[-1]-1), asarray(Exy.eval([x_test, y_test])), delimiter=',')

for i in range(l): 
    print(ht[i].history.keys())

#saving converged Lame parameters
lmbd_final = [0]*l;
mu_final = [0]*l;
for i in range(l):
    lmbd_final[i] = ht[i].history['lmbd']
    mu_final[i] = ht[i].history['mu']
savetxt('lambda_{}X{}.csv'.format(Nl[0]-1,Nl[-1]-1), asarray(lmbd_final)[0], delimiter=',')
savetxt('mu_{}X{}.csv'.format(Nl[0]-1,Nl[-1]-1), asarray(mu_final)[0], delimiter=',')

#plotting loss 
loss = [0]*l;
for i in range(l):
    plt.figure()
    plt.semilogy(ht[i].history['loss'])
    plt.xlabel("Number of Epochs")
    plt.ylabel(r"Total loss, $\mathcal{L}$")
    plt.title("Total loss vs. number of epochs ({}X{})".format(Nl[i]-1,Nl[i]-1))
    plt.savefig('loss {}X{} grid.png'.format(Nl[i]-1,Nl[i]-1))
    loss[i] = (ht[i].history['loss']);
print(loss)
#saving loss
savetxt('loss {}X{}.csv'.format(Nl[0]-1,Nl[-1]-1), asarray(loss), delimiter=',')

#analytical solutions 
u_act = np.cos(2*pi*x_test) * np.sin(pi*y_test)
v_act = np.sin(pi*x_test) * Q * y_test**4/4
Exx_act = -2*pi*np.sin(2*pi*x_test) * np.sin(pi*y_test)
Eyy_act = 4.0*y_test**3*np.sin(pi*x_test)
Exy_act = 0.5*pi*y_test**4*np.cos(pi*x_test) + 0.5*pi*np.cos(2*pi*x_test)*np.cos(pi*y_test)
Sxx_act = 4.0*y_test**3*np.sin(pi*x_test) - 4.0*pi*np.sin(2*pi*x_test)*np.sin(pi*y_test)
Syy_act = 8.0*y_test**3*np.sin(pi*x_test) - 2.0*pi*np.sin(2*pi*x_test)*np.sin(pi*y_test)
Sxy_act = 0.5*pi*y_test**4*np.cos(pi*x_test) + 0.5*pi*np.cos(2*pi*x_test)*np.cos(pi*y_test)
#net displacement
uv_act = np.sqrt(u_act**2 + v_act**2)

#Calculation of absolute, l2-norm and energy-norm errors
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
k=0
for N in Nl:
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
    #L2 norm error in displacement vectors
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
    print("L2 norm error in displacement vectors with {}X{} grid =".format(N-1,N-1), uv_err_l2_norm[k])

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
    print("L2 norm error in Stresses with {}X{} grid =".format(N-1,N-1), S_err_l2_norm[k])    
    
    #Error with energy norm calculations
    #Error with energy norm in displacement vectors
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
    print("Energy norm error in displacement vectors with {}X{} grid =".format(N-1,N-1), uv_err_energy_norm[k])
    
    #storing errors in lists
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
        r'$||e(|u|)||_{L^2}$':uv_err_point,
        r'$||e(\sigma)||_{L^2}$': S_err_point,
    })
    energy_err_data.append({
        r'$||e(|u|)||_{A}$ (absolute)':energy_uv_err_point, 
        r'$||e(|u|)||_{A}$ (relative)':en_rel_uv_err_point
    })
    
    k=k+1;
print(type(uv_err_l2_norm))
#saving L2 error norm of net displacement
savetxt('L2 Error_norm_disp {}X{}.csv'.format(Nl[0]-1,Nl[-1]-1), asarray(uv_err_l2_norm), delimiter=',')
#saving L2 error norm of stresses
savetxt('L2 Error_norm_Stress {}X{}.csv'.format(Nl[0]-1,Nl[-1]-1), asarray(S_err_l2_norm), delimiter=',')
#saving energy error norm of displacements
savetxt('Energy Error_norm {}X{}.csv'.format(Nl[0]-1,Nl[-1]-1), uv_err_energy_norm, delimiter=',')
print(type(uv_err_l2_norm))

#calculating convergence ratio (maximum net displacement in pinn over analytical solution)
uv_conv_ratio = []
for index,N in enumerate(Nl):
    uv_act_max = np.amax(abs((uv_act)))
    uv_act_i = np.argmax(abs((uv_act)))
    if (uv_act_i+1)%Nt!=0:
        uv_act_max_row = int((uv_act_i+1)/Nt)
        uv_act_max_col = ((uv_act_i+1)%Nt)-1
    else:
        uv_act_max_row = int((uv_act_i+1)/Nt)-1
        uv_act_max_col = Nt-1
    uv_test_max = (np.sqrt(test_data[index][r'${u_x}$'][uv_act_max_row][uv_act_max_col]**2+
                                    test_data[index][r'${u_y}$'][uv_act_max_row][uv_act_max_col]**2))
    uv_conv_ratio.append(np.abs(uv_test_max-uv_act_max)/uv_act_max)
print("Convergence ratio in uv: {}".format(uv_conv_ratio))
savetxt('Convergence ratio in uv {}-{}.csv'.format(Nl[0]-1,Nl[-1]-1), asarray(uv_conv_ratio), delimiter=',')

#storing analytical solution in a list
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
    plt.savefig('test output {}X{} grid.png'.format(N-1,N-1))

#contour plots for analytical solution
def cust_plot_actual(ax, val, label):
    im = ax.pcolor(x_test, y_test, val, cmap='seismic',
                  vmin=-np.abs(val).max(), vmax=np.abs(val).max(),shading='auto')
    ax.set_title(label, fontsize=30, weight='bold')
    ax.axes.xaxis.set_ticklabels([]) 
    cb = plt.colorbar(im, ax=ax)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(20)
    
fig, (ax1,ax2) = plt.subplots(2,4,figsize=(25,12), sharey=True)
j=0
for i, (key, val) in enumerate(actual_data.items()):
    if i<4:
        cust_plot_actual(ax1[i], val, key)
    else:
        cust_plot_actual(ax2[j], val, key)
        j=j+1
    
plt.tight_layout()
plt.savefig('actual output.png'.format(N-1,N-1))

#contour plot for absolute errors
for index,N in enumerate(Nl):
    def cust_plot_error(ax, val, label,N):
        im = ax.pcolor(x_test, y_test, val, cmap='seismic', 
                      vmin=np.abs(val).min(), vmax=np.abs(val).max(),shading='auto')
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
    plt.savefig('abs error {}X{} grid.png'.format(N-1,N-1))
    #plt.show()

#contour plot for l2-norm errors
for index,N in enumerate(Nl):
    def cust_plot_error(ax, val, label,N):
        im = ax.pcolor(x_test, y_test, val, cmap='seismic', 
                      vmin=np.abs(val).min(), vmax=np.abs(val).max(),shading='auto')
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
    plt.savefig('l2 error {}X{} grid.png'.format(N-1,N-1))
    #plt.show()

#contour plot for energy-norm errors
for index,N in enumerate(Nl):
    def cust_plot_error(ax, val, label,N):
        im = ax.pcolor(x_test, y_test, val, cmap='seismic', 
                      vmin=np.abs(val).min(), vmax=np.abs(val).max(),shading='auto')
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



