import h5py as h5
import functions as fn
import numpy as np
from matplotlib import pyplot as plt
import sys
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
# filename = 'gk_coulomb_NFLR_6_P_4_J_2_N_50_kpm_4.0.h5'
filename = 'gk_landau_P10_J5_dk_5e-2_km_2.0_NFLR_12.h5'

if len(sys.argv) >= 2:
    filename = sys.argv[1]

file = h5.File(filename)

# Dump the content hierarchy
fn.dump_h5(file)

# Select data for corresponding kperp
kperp = 0
data = file['%05i' % kperp]

# Select the matrix for electrons-electrons collisions
matrix_ee = np.array(data['Caapj']['Ceepj'])
matrix_ii = np.array(data['Caapj']['Ciipj'])

# Decimate threshold

# matrix_ee[abs(matrix_ee) <= 0.00] = np.nan
# matrix_ii[abs(matrix_ii) <= 0.00] = np.nan

# Get number of hermite-laguerre for electrons
dims_e = np.array(file['dims_e'])+np.array([1,1])
dims_i = np.array(file['dims_e'])+np.array([1,1])


# Decimate threshold
matrix_ee[abs(matrix_ee) <= 0.01] = 0
matrix_ii[abs(matrix_ii) <= 0.01] = 0

# Compute the correlation matrix
correlation_matrix_ee = np.corrcoef(matrix_ee)
correlation_matrix_ii = np.corrcoef(matrix_ii)

correlation_matrix_ee[abs(matrix_ee) <= 0.01] = 0
correlation_matrix_ii[abs(matrix_ii) <= 0.01] = 0

dens_ee = np.sum(correlation_matrix_ee, axis=0 )

plt.plot(np.linspace(0,66,66),dens_ee)

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram Plot')

# Show the plot
plt.show()


dens_ii = np.sum(correlation_matrix_ii, axis=0 )

plt.plot(np.linspace(0,66,66),dens_ii)

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram Plot')

# Show the plot
plt.show()


