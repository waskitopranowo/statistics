import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from scipy import interpolate
from scipy.stats import norm
import gstools as gs

# MGK for qualitative data

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

data_unique, counts = np.unique(value,return_counts=True)
data_sort = np.sort(data_unique)
epdf = np.zeros((len(data_sort)))

plt.bar(range(len(data_unique)), counts/len(value), align='center')
plt.xticks(range(len(data_unique)), data_unique)
plt.ylabel('Frequency')
plt.xlabel('Risk Level')

for i in np.arange(0,len(data_sort),1):
    epdf[i] = len(np.where(value == data_sort[i])[0])/len(value)
ecdf = np.cumsum(epdf)

z = np.linspace(-3,3,101)
cdf = norm.cdf(z, loc=0, scale=1)
f2 = interpolate.interp1d(cdf, z)
range_bound = f2(ecdf[0:len(ecdf)-1])
range_bound = np.hstack((-3, range_bound, 3))

nst = np.zeros((len(value)))
for i in range(len(nst)):
    j = np.where(value[i] == data_sort)[0]
    # nst[i] = np.random.rand(1) * (range_bound[j + 1] - range_bound[j]) + range_bound[j]
    nst[i] = 0.5 * (range_bound[j + 1] - range_bound[j]) + range_bound[j]

bins = np.linspace(0, 6000, 6)
bin_center, gamma = gs.vario_estimate_unstructured([xdata, ydata], nst, bins)
# model = gs.Gaussian(dim=1, var=0.26, len_scale=2000)    #len_scale=range, var:sill
# model = gs.Spherical(dim=1, var=0.25, len_scale=3500)    #len_scale=range, var:sill
model = gs.Spherical(dim=1, var=2, len_scale=4000)    #len_scale=range, var:sill
model.plot(x_max=6000)
plt.plot(bin_center, gamma, '.'),
plt.xlabel('Lag'), plt.ylabel('Variogram, $\gamma$(L)')
plt.show()

pk_kwargs = model.pykrige_kwargs
OK = OrdinaryKriging(xdata, ydata, nst, **pk_kwargs)
nst_est, evar = OK.execute("grid", x, y)

est_val = np.zeros_like(nst_est)
for i in range(len(data_sort)):
    est_val[(nst_est < range_bound[i + 1]) & (nst_est >= range_bound[i])] = i

fig, axs = plt.subplots()
caxs = axs.imshow(est_val, aspect='auto', origin='lower', extent=[Xmin, Xmax, Ymin, Ymax], cmap='YlOrRd')
cb = fig.colorbar(caxs, ticks=np.arange(0,len(data_sort), 1), orientation="horizontal")
cb.ax.set_xticklabels(data_sort)
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.show()

# data_sort = np.sort(value)
# ecdf = np.arange(1,len(data_sort)+1,1)/len(data_sort)
#
# z = np.linspace(-3, 3, 101)
# cdf = norm.cdf(z, loc=0, scale=1)
# f = interpolate.interp1d(cdf, z)
# nst = f(ecdf[0:len(ecdf)-1])
# nst = np.hstack((nst, 3))
#
# pk_kwargs = model.pykrige_kwargs
# OK1 = OrdinaryKriging(xdata, ydata, nst, **pk_kwargs)
# z1, ss1 = OK1.execute("grid", x, y)
#
# f = interpolate.interp1d(z, cdf)
# z1f = np.asarray(z1.flatten())
# cdf2 = f(z1f)
# f2 = interpolate.interp1d(ecdf, data_sort)
# z1 = f2(cdf2)
# z1 = z1.reshape(ny,nx)
#
# plt.imshow(z1, aspect='auto', origin='lower')
# plt.colorbar()
# plt.show()
