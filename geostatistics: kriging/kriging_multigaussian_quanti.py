import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from scipy import interpolate
from scipy.stats import norm
import gstools as gs

# MGK for quantative data
# Normal Score Transform by using Empirical CDF

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

data_sort = np.sort(value)
ecdf = np.arange(1,len(data_sort)+1,1)/len(data_sort)
f = interpolate.interp1d(data_sort, ecdf)
data_norm = f(value)

z = np.linspace(-3, 3, 101)
cdf = norm.cdf(z, loc=0, scale=1)
f = interpolate.interp1d(cdf, z)
data_norm[data_norm > np.max(cdf)] = np.max(cdf) # to avoid cdf > 1
nst = f(data_norm)

bins = np.linspace(0, 4000, 8)
bin_center, gamma = gs.vario_estimate_unstructured([xdata, ydata], nst, bins)
model = gs.Spherical(dim=1, var=2.2, len_scale=2300)
model.plot(x_max=4000)
plt.plot(bin_center, gamma, '.'),
plt.xlabel('Lag'), plt.ylabel('Variogram, $\gamma$(L)')
plt.show()

pk_kwargs = model.pykrige_kwargs
OK = OrdinaryKriging(xdata, ydata, nst, **pk_kwargs)
nst_est, evar = OK.execute("grid", x, y)
nst_est[nst_est > 3] = 3
nst_est[nst_est < -3] = -3

f = interpolate.interp1d(z, cdf)
nst_est = np.asarray(nst_est.flatten())
cdf2 = f(nst_est)
f2 = interpolate.interp1d(ecdf, data_sort)
val_est = f2(cdf2)

val_est = val_est.reshape(ny,nx)

fig, axs = plt.subplots()
caxs = axs.imshow(val_est, aspect='auto', interpolation='bilinear', origin='lower', extent=[Xmin, Xmax, Ymin, Ymax])
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
fig.colorbar(caxs, orientation="horizontal")
plt.show()
