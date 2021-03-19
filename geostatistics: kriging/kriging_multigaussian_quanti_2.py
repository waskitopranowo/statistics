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

num_bins = 1 + int(3.3*np.log10(len(value)))
bin_bd = np.linspace(np.min(value)*0.9999,np.max(value)*1.0001, num_bins+1)
n = np.zeros((num_bins))
for i in range(num_bins):
    n[i] = len(value[(value>bin_bd[i]) & (value<=bin_bd[i+1])])
dbin = bin_bd[1] - bin_bd[0]
bin_c = bin_bd[0:len(bin_bd)-1] + dbin/2
bin_c = np.hstack((bin_c[0]-dbin, bin_c, bin_c[len(bin_c)-1]+dbin))
n = np.hstack((0, n, 0))/len(value)
ncdf = np.cumsum(n)

f = interpolate.interp1d(bin_c, ncdf)
data_norm = f(value)

z = np.linspace(-3, 3, 101)
cdf = norm.cdf(z, loc=0, scale=1)
f = interpolate.interp1d(cdf, z)
data_norm[data_norm>np.max(cdf)] = np.max(cdf)
nst = f(data_norm)
# plt.plot(data_norm)
# plt.show()

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
f2 = interpolate.interp1d(ncdf, bin_c)
val_est = f2(cdf2)

val_est = val_est.reshape(ny, nx)

fig, axs = plt.subplots()
caxs = axs.imshow(val_est, aspect='auto', interpolation='bilinear', origin='lower', extent=[Xmin, Xmax, Ymin, Ymax])
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
fig.colorbar(caxs, orientation="horizontal")
plt.show()
