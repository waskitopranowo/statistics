import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import gstools as gs

# IK for qualitative data

data = np.loadtxt('Data_Qual.txt', skiprows=1, dtype=str)
xdata = data[:, 0].astype('float')
ydata = data[:, 1].astype('float')
value = data[:, 2]

Xmin = np.min(xdata)
Xmax = np.max(xdata)
Ymin = np.min(ydata)
Ymax = np.max(ydata)

nx = 100
ny = 100

x = np.linspace(Xmin, Xmax, nx)
y = np.linspace(Ymin, Ymax, ny)

I = np.zeros((len(value)))

thres = 'D.High'
I[value == thres] = 1

bins = np.linspace(0, 6000, 8)
bin_center, gamma = gs.vario_estimate_unstructured([xdata, ydata], I, bins)
# model = gs.Gaussian(dim=1, var=0.26, len_scale=2000)    #len_scale=range, var:sill
# model = gs.Spherical(dim=1, var=0.25, len_scale=3500)    #len_scale=range, var:sill
model = gs.Spherical(dim=1, var=0.19, len_scale=2400)    #len_scale=range, var:sill
model.plot(x_max=6000)
plt.plot(bin_center, gamma, '.'),
plt.xlabel('Lag'), plt.ylabel('Variogram, $\gamma$(L)')
plt.legend(['Spherical variogram','Calculated variogram'])
plt.show()

pk_kwargs = model.pykrige_kwargs
OK = OrdinaryKriging(xdata, ydata, I, **pk_kwargs)
val_est, evar = OK.execute("grid", x, y)
val_est = (val_est - np.min(val_est))/(np.max(val_est) - np.min(val_est))

# plt.figure(1)
fig, axs = plt.subplots()
caxs = axs.imshow(val_est, aspect='auto', interpolation='bilinear', origin='lower', extent=[Xmin, Xmax, Ymin, Ymax], cmap='gist_yarg')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
fig.colorbar(caxs, orientation="horizontal")
plt.show()
