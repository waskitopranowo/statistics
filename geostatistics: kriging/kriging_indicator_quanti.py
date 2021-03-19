import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import gstools as gs

# IK for quantative data

data = np.loadtxt('Data_Quan.txt', skiprows=1)
xdata = data[:, 0]
ydata = data[:, 1]
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
thres = 0.1

I[value >= thres] = 1
I[value < thres] = 0

bins = np.linspace(0, 4000, 6)
bin_center, gamma = gs.vario_estimate_unstructured([xdata, ydata], I, bins)
model = gs.Spherical(dim=1, var=0.3, len_scale=2500)    #len_scale=range, var:sill
model.plot(x_max=4000)
plt.plot(bin_center, gamma, '.'),
plt.xlabel('Lag'), plt.ylabel('Variogram, $\gamma$(L)')
plt.show()

pk_kwargs = model.pykrige_kwargs
OK = OrdinaryKriging(xdata, ydata, I, **pk_kwargs)
val_est, evar = OK.execute("grid", x, y)
val_est = (val_est - np.min(val_est))/(np.max(val_est) - np.min(val_est))

fig, axs = plt.subplots()
caxs = axs.imshow(val_est, aspect='auto', interpolation='bilinear', origin='lower', extent=[Xmin, Xmax, Ymin, Ymax], cmap='gist_yarg')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
fig.colorbar(caxs, orientation="horizontal")
plt.show()
