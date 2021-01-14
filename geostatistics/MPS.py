'''Multiple-point simulation with Direct Sampling Algorithm by using Python 

This code is modified and translated to python code by Author in 2020 from Gregoire Mariethoz (2009)
'''

import numpy as np
import matplotlib.pyplot as plt


# Import Training Image
ti=np.loadtxt('sungaibiner.txt')
plt.title('Training Image')
plt.imshow(ti, cmap='gray', origin='lower')
plt.show()

baris=ti.shape[0]
kolom=ti.shape[1]

# Parameters are:
simul_size = [80, 80]      #size of the simulation: [y x]
ti_size = [baris, kolom]   #size of the ti: [y x]
template = [30, 30]          #size of the template: [y x]
fract_of_ti_to_scan = 0.03          #maximum fraction of the TI to scan
thr = 0.01                         #threshold (between 0 and 1)

x = np.arange(0, ti_size[0], 1)
y = np.arange(0, ti_size[1], 1)

# DS Simulation
#defining shifts related to the template size
yshift = int(np.floor(template[0]/2))
xshift = int(np.floor(template[1]/2))

#load and creating simulation space with a wrapping of NaNs of the size "shift"
data=np.loadtxt('hasilcuplik.txt')
plt.title('Sample data')
plt.imshow(data, cmap='jet', origin='lower')
plt.colorbar()
plt.show()

simulv=np.zeros((yshift,len(data)))
simulv+=np.nan
simulh=np.zeros((len(data)+xshift*2,xshift))
simulh+=np.nan

vstack=np.vstack((simulv,data,simulv))
cuplik=np.hstack((simulh,vstack,simulh))

#reducing the size of the ti to avoid scanning outside of it
ti_size[0] = int(ti.shape[0]-2*yshift)
ti_size[1] = int(ti.shape[1]-2*yshift)

#creating simulation space with a wrapping of NaNs of the size "shift" for summing the result of iteration
A = int(simul_size[0]+2*yshift)
B = int(simul_size[1]+2*xshift)
simul_itt= np.zeros((A, B))


#Define size of simulation and iteration
sizesim = int(simul_size[0]*simul_size[1])
n_iteration=25

for it in range(n_iteration):

   #defining path in ti and in simulation
   path_ti = np.random.permutation(int(ti_size[0] * ti_size[1]))
   path_sim = np.random.permutation(int(simul_size[0] * simul_size[1]))

   tinod = -1
   progress = 0

   print('Iterasi ke %s'%(it+1))
   simul = cuplik

   #Define dummy space for each iteration
   dummy = np.zeros((A, B))

   for simnod in np.arange(0,sizesim, 1):
       #find note in the simulation grid
       xsim = int(np.floor(path_sim[simnod] / simul_size[0]))
       ysim = int(path_sim[simnod] - (xsim * simul_size[0]))

       #define the point of shifting it to avoid scanning NaNs
       point_sim = [ysim + yshift, xsim + xshift]

       data_event_sim = simul[point_sim[0] - yshift : point_sim[0] + yshift + 1,
                        point_sim[1] - xshift: point_sim[1] + xshift + 1]

       mindist = 1
       tries = -1
       max_scan = (np.size(path_ti)) * fract_of_ti_to_scan

       #reducing the data event to its informed nodes
       no_data_indicator = np.isfinite(data_event_sim)
       no_idx = np.asarray(np.where(np.isfinite(data_event_sim)))
       data_event_sim = data_event_sim[no_data_indicator]


       while 1 == 1:
           tinod = tinod + 1
           tries = tries + 1
           if tinod > len(path_ti)-1:
               tinod = 0
           progress_current = np.ceil((simnod * 100) / sizesim)
           if progress_current > progress:
               progress = progress_current
               print(progress, '% completed')

           xti = int(np.floor(path_ti[tinod] / ti_size[0]))
           yti = int(path_ti[tinod] - (xti * ti_size[0]))

           #scanned
           point_ti = [yti + yshift, xti + xshift]
           data_event_ti = ti[point_ti[0] - yshift:point_ti[0] + yshift + 1,
                           point_ti[1] - xshift: point_ti[1] + xshift + 1]


           if np.sum(np.sum(no_data_indicator,axis=0)) == 0:
               simul[point_sim[0], point_sim[1]] = ti[point_ti[0], point_ti[1]]
               break
           data_event_ti = data_event_ti[no_data_indicator]
           distance = np.mean(np.sign(abs(data_event_sim - data_event_ti)))

           if distance <= thr:
               simul[point_sim[0], point_sim[1]] = ti[point_ti[0], point_ti[1]]
               break
           else:
               if distance < mindist:
                   mindist = distance
                   bestpoint = point_ti

               if tries > max_scan:
                   simul[point_sim[0], point_sim[1]] = ti[bestpoint[0], bestpoint[1]]
                   break
   dummy[:] =simul
   simul_itt+=dummy
   plt.title('Simulation Iteration %s'%(it+1))
   plt.imshow(simul, aspect='auto',cmap='gray',origin='lower')
   plt.show()
   print('='*25)

#Averaging the result of simulation
simul_itt = simul_itt[yshift:(-1*yshift),xshift:(-1*xshift)]
simul_itt=simul_itt/n_iteration

#Plot TI and Simulation
plt.subplot(1, 2, 1)
plt.title('Training Image')
plt.imshow(ti, aspect='auto',cmap='gray', origin ='lower')

plt.subplot(1,2,2)
plt.title('Simulation with %s Iteration'%n_iteration)
plt.imshow(simul_itt,cmap='jet', origin ='lower', aspect='auto')
plt.colorbar()
plt.show()
