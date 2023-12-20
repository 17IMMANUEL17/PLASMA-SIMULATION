import h5py as h5
import functions as fn
import numpy as np
from matplotlib import pyplot as plt
import sys

#filename = 'gk_landau_P10_J5_dk_5e-2_km_2.0_NFLR_12.h5'
#filename = 'dk.coulomb.ab.P6J3/ei.h5'
filename = 'out/dkcoulomb.self.20.10/ei.h5'

if len(sys.argv) >= 2:
    filename = sys.argv[1]

file = h5.File(filename)

# Dump the content hierarchy
fn.dump_h5(file)

# Select data for corresponding kperp
kperp = 0
#data = file['%05i' % kperp]
data = file

# Select the matrix for electrons-electrons collisions
matrix_ee = np.array(data['Ceipj']['CeipjT'])
matrix_ii = np.array(data['Ceipj']['CeipjT'])

# Decimate threshold
# #matrix_ee[abs(matrix_ee) <= 0.00] = np.nan
# matrix_ii[abs(matrix_ii) <= 0.00] = np.nan

# Get number of hermite-laguerre for electrons
#dims_e = np.array(file['dims_e'])+np.array([1,1])
#dims_i = np.array(file['dims_e'])+np.array([1,1])
dims_e = np.array(data['Ceipj']['CeipjT']['Pmaxe'])
dims_i = np.array(data['Ceipj']['CeipjT']['Pmaxi'])

# Print the matrix
plt.figure()
plt.pcolor(matrix_ee)
plt.title('Collision Electrons-electrons P=%i, J=%i' % (dims_e,dims_i))
plt.xlabel('(p,j)')
plt.ylabel('(p,j)')
plt.gca().invert_yaxis()
plt.colorbar()
plt.savefig('matrix_ee.eps', format = 'eps')

plt.figure()
plt.pcolor(matrix_ii)
plt.title('Collision Ions-ions P=%i, J=%i') #% (dims_i[0],dims_i[1]))
plt.xlabel('(p,j)')
plt.ylabel('(p,j)')
plt.gca().invert_yaxis()
plt.colorbar()

plt.show()

#no dependence between odd ps and even ps; we can create 2 subsystems reducing computational complexity but we lose structure