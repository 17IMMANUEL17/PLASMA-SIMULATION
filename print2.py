import h5py as h5
import functions as fn
import numpy as np
from matplotlib import pyplot as plt
import sys
from scipy.stats import linregress

#filename = 'gk_landau_P10_J5_dk_5e-2_km_2.0_NFLR_12.h5'
#filename = 'dk.coulomb.ab.P6J3/ei.h5'

filename = 'C:\\Users\\anton\\Downloads\\out\\newP22J12_real\\fields.dat.h5'



file = h5.File(filename, 'r')

# Dump the content hierarchy
#fn.dump_h5(file)

data = file['data']['var1d']['phi']
# /data/var1d/phi/coordz
time =  np.array(file['data']['var1d']['time'])
# tmp = np.array(data['000005'])


#fn.dump_h5(data['000005'])
elems_p = []
for i in range(len(time)):
    vals = []
    for cplx in data['%06d'%i]:
        value = cplx[0] + 1j*cplx[1]
        vals.append(value)
    elem = vals[1]
    elems_p.append(abs(elem))

elems_p = np.array(elems_p)
mask = elems_p > 1e-5


#plt.semilogy(time,elems_p, label = 'Coulomb')



# filename = 'C:\\Users\\anton\\Downloads\\out\\newP22J12_app_glob_50\\fields.dat.h5'



# file = h5.File(filename, 'r')

# # Dump the content hierarchy
# #fn.dump_h5(file)

# data = file['data']['var1d']['phi']
# # /data/var1d/phi/coordz
# time =  np.array(file['data']['var1d']['time'])
# # tmp = np.array(data['000005'])

# #fn.dump_h5(data['000005'])
# elems = []
# for i in range(len(time)):
#     vals = []
#     for cplx in data['%06d'%i]:
#         value = cplx[0] + 1j*cplx[1]
#         vals.append(value)
#     elem = vals[1]
#     elems.append(abs(elem))
# print(time.shape)

# plt.figure()
# #plt.plot(np.array(vals)**2)
# plt.semilogy(time[mask] ,abs( np.array(elems)[mask] - elems_p[mask])/abs(elems_p[mask]), label = 'glob50')



# filename = 'C:\\Users\\anton\\Downloads\\out\\newP22J12_app_glob_70\\fields.dat.h5'


# file = h5.File(filename, 'r')

# # Dump the content hierarchy
# #fn.dump_h5(file)

# data = file['data']['var1d']['phi']
# # /data/var1d/phi/coordz
# time =  np.array(file['data']['var1d']['time'])
# # tmp = np.array(data['000005'])


# #fn.dump_h5(data['000005'])
# elems = []
# for i in range(len(time)):
#     vals = []
#     for cplx in data['%06d'%i]:
#         value = cplx[0] + 1j*cplx[1]
#         vals.append(value)
#     elem = vals[1]
#     elems.append(abs(elem))

# print(time.shape)
# #plt.figure()
# #plt.plot(np.array(vals)**2)
# plt.semilogy(time[mask],abs( np.array(elems)[mask] - elems_p[mask])/abs(elems_p[mask]), label = 'glob70')


filename = 'C:\\Users\\anton\\Downloads\\out\\newP22J12_app_glob_90\\fields.dat.h5'



file = h5.File(filename, 'r')

#Dump the content hierarchy
fn.dump_h5(file)

data = file['data']['var1d']['phi']
#/data/var1d/phi/coordz
time =  np.array(file['data']['var1d']['time'])
tmp = np.array(data['000005'])


fn.dump_h5(data['000005'])
elems = []
for i in range(len(time)):
    vals = []
    for cplx in data['%06d'%i]:
        value = cplx[0] + 1j*cplx[1]
        vals.append(value)
    elem = vals[1]
    elems.append(abs(elem))


plt.figure()
plt.plot(np.array(vals)**2)
plt.semilogy(time[mask],abs( np.array(elems)[mask] - elems_p[mask])/abs(elems_p[mask]), label = 'glob90')


filename = 'C:\\Users\\anton\\Downloads\\out\\newP22J12_app_glob_101\\fields.dat.h5'



file = h5.File(filename, 'r')

#Dump the content hierarchy
fn.dump_h5(file)

data = file['data']['var1d']['phi']
#/data/var1d/phi/coordz
time =  np.array(file['data']['var1d']['time'])
tmp = np.array(data['000005'])


fn.dump_h5(data['000005'])
elems = []
for i in range(len(time)):
    vals = []
    for cplx in data['%06d'%i]:
        value = cplx[0] + 1j*cplx[1]
        vals.append(value)
    elem = vals[1]
    elems.append(abs(elem))


plt.plot(np.array(vals)**2)
plt.semilogy(time[mask],abs( np.array(elems)[mask] - elems_p[mask])/abs(elems_p[mask]), label = 'glob101')


filename = 'C:\\Users\\anton\\Downloads\\out\\newP22J12_app_glob_120\\fields.dat.h5'



file = h5.File(filename, 'r')

#Dump the content hierarchy
fn.dump_h5(file)

data = file['data']['var1d']['phi']
#/data/var1d/phi/coordz
time =  np.array(file['data']['var1d']['time'])
tmp = np.array(data['000005'])


fn.dump_h5(data['000005'])
elems = []
for i in range(len(time)):
    vals = []
    for cplx in data['%06d'%i]:
        value = cplx[0] + 1j*cplx[1]
        vals.append(value)
    elem = vals[1]
    elems.append(abs(elem))


plt.plot(np.array(vals)**2)
plt.semilogy(time[mask],abs( np.array(elems)[mask] - elems_p[mask])/abs(elems_p[mask]), label = 'glob120')


filename = 'C:\\Users\\anton\\Downloads\\out\\newP22J12_app_glob_125\\fields.dat.h5'

file = h5.File(filename, 'r')

#Dump the content hierarchy
fn.dump_h5(file)

data = file['data']['var1d']['phi']
#/data/var1d/phi/coordz
time =  np.array(file['data']['var1d']['time'])
tmp = np.array(data['000005'])


fn.dump_h5(data['000005'])
elems = []
for i in range(len(time)):
    vals = []
    for cplx in data['%06d'%i]:
        value = cplx[0] + 1j*cplx[1]
        vals.append(value)
    elem = vals[1]
    elems.append(abs(elem))



plt.plot(np.array(vals)**2)
plt.semilogy(time[mask],abs( np.array(elems)[mask] - elems_p[mask])/abs(elems_p[mask]), label = 'glob125')


filename = 'C:\\Users\\anton\\Downloads\\out\\newP22J12_app_glob_130\\fields.dat.h5'


file = h5.File(filename, 'r')

# Dump the content hierarchy
#fn.dump_h5(file)

data = file['data']['var1d']['phi']
# /data/var1d/phi/coordz
time =  np.array(file['data']['var1d']['time'])
# tmp = np.array(data['000005'])


#fn.dump_h5(data['000005'])
elems = []
for i in range(len(time)):
    vals = []
    for cplx in data['%06d'%i]:
        value = cplx[0] + 1j*cplx[1]
        vals.append(value)
    elem = vals[1]
    elems.append(abs(elem))


#plt.figure()
#plt.plot(np.array(vals)**2)
plt.semilogy(time[mask],abs( np.array(elems)[mask] - elems_p[mask])/abs(elems_p[mask]), label = 'glob130')


filename = 'C:\\Users\\anton\\Downloads\\out\\newP22J12_app_glob_150\\fields.dat.h5'

file = h5.File(filename, 'r')

# Dump the content hierarchy
#fn.dump_h5(file)

data = file['data']['var1d']['phi']
# /data/var1d/phi/coordz
time =  np.array(file['data']['var1d']['time'])
# tmp = np.array(data['000005'])


#fn.dump_h5(data['000005'])
elems = []
for i in range(len(time)):
    vals = []
    for cplx in data['%06d'%i]:
        value = cplx[0] + 1j*cplx[1]
        vals.append(value)
    elem = vals[1]
    elems.append(abs(elem))


#plt.figure()
#plt.plot(np.array(vals)**2)
plt.semilogy(time[mask],abs( np.array(elems)[mask] - elems_p[mask])/abs(elems_p[mask]), label = 'glob150')


filename = 'C:\\Users\\anton\\Downloads\\out\\newP22J12_app_glob_170\\fields.dat.h5'


file = h5.File(filename, 'r')

# Dump the content hierarchy
#fn.dump_h5(file)

data = file['data']['var1d']['phi']
# /data/var1d/phi/coordz
time =  np.array(file['data']['var1d']['time'])
# tmp = np.array(data['000005'])


#fn.dump_h5(data['000005'])
elems = []
for i in range(len(time)):
    vals = []
    for cplx in data['%06d'%i]:
        value = cplx[0] + 1j*cplx[1]
        vals.append(value)
    elem = vals[1]
    elems.append(abs(elem))


#plt.figure()
#plt.plot(np.array(vals)**2)
plt.semilogy(time[mask],abs( np.array(elems)[mask] - elems_p[mask])/abs(elems_p[mask]), label = 'glob170')


filename = 'C:\\Users\\anton\\Downloads\\out\\newP22J12_app_glob_200\\fields.dat.h5'


file = h5.File(filename, 'r')

# Dump the content hierarchy
#fn.dump_h5(file)

data = file['data']['var1d']['phi']
# /data/var1d/phi/coordz
time =  np.array(file['data']['var1d']['time'])
# tmp = np.array(data['000005'])


#fn.dump_h5(data['000005'])
elems = []
for i in range(len(time)):
    vals = []
    for cplx in data['%06d'%i]:
        value = cplx[0] + 1j*cplx[1]
        vals.append(value)
    elem = vals[1]
    elems.append(abs(elem))


#plt.figure()
#plt.plot(np.array(vals)**2)
plt.semilogy(time[mask],abs( np.array(elems)[mask] - elems_p[mask])/abs(elems_p[mask]), label = 'glob200')


filename = 'C:\\Users\\anton\\Downloads\\out\\newP22J12_app_glob_250\\fields.dat.h5'


file = h5.File(filename, 'r')

# Dump the content hierarchy
#fn.dump_h5(file)

data = file['data']['var1d']['phi']
# /data/var1d/phi/coordz
time =  np.array(file['data']['var1d']['time'])
# tmp = np.array(data['000005'])


#fn.dump_h5(data['000005'])
elems = []
for i in range(len(time)):
    vals = []
    for cplx in data['%06d'%i]:
        value = cplx[0] + 1j*cplx[1]
        vals.append(value)
    elem = vals[1]
    elems.append(abs(elem))


#plt.figure()
#plt.plot(np.array(vals)**2)
plt.semilogy(time[mask],abs( np.array(elems)[mask] - elems_p[mask])/abs(elems_p[mask]), label = 'glob250')

plt.legend()
plt.show()