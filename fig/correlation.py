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
matrix_ee[abs(matrix_ee) <= 0.00] = 0
matrix_ii[abs(matrix_ii) <= 0.00] = 0

# Compute the correlation matrix
correlation_matrix_ee = np.corrcoef(matrix_ee)
correlation_matrix_ii = np.corrcoef(matrix_ii)

correlation_matrix_ee[abs(matrix_ee) <= 0.01] = np.nan
correlation_matrix_ii[abs(matrix_ii) <= 0.01] = np.nan
# Print the matrix
# plt.figure()
# plt.pcolor(abs(correlation_matrix_ee))
# plt.title('Collision Electrons-electrons P=%i, J=%i' % (dims_e[0],dims_e[1]))
# plt.xlabel('(p,j)')
# plt.ylabel('(p,j)')
# plt.gca().invert_yaxis()
# plt.colorbar()

# plt.figure()
# plt.pcolor(abs(correlation_matrix_ii))
# plt.title('Collision Ions-ions P=%i, J=%i' % (dims_i[0],dims_i[1]))
# plt.xlabel('(p,j)')
# plt.ylabel('(p,j)')
# plt.gca().invert_yaxis()
# plt.colorbar()

# plt.show()

# rank = 63

# sparse_matrix_ee = csr_matrix(matrix_ee)
# #sparse_matrix_ii = csr_matrix(matrix_ii)
# svd_ee = TruncatedSVD(n_components=rank)
# svd_ii = TruncatedSVD(n_components=rank)
# approximated_matrix_ee = svd_ee.fit_transform(sparse_matrix_ee)
# #approximated_matrix_ii = svd_ii.fit_transform(sparse_matrix_ii)
# approximated_matrix_ee[abs(approximated_matrix_ee) <= 0.05] = np.nan
# #approximated_matrix_ii[abs(approximated_matrix_ii) <= 0.05] = np.nan
# print(np.linalg.matrix_rank(matrix_ee))
# print(np.linalg.matrix_rank(matrix_ii))
# U, Sigma1, VT = np.linalg.svd(matrix_ee)
# U, Sigma2, VT = np.linalg.svd(matrix_ii)
# print(Sigma1)
# print(Sigma2)

# plt.figure()
# plt.pcolor(approximated_matrix_ee)
# plt.title('Collision Ions-ions P=%i, J=%i' % (dims_i[0],dims_i[1]))
# plt.xlabel('(p,j)')
# plt.ylabel('(p,j)')
# plt.gca().invert_yaxis()
# plt.colorbar()

# plt.figure()
# plt.pcolor(approximated_matrix_ii)
# plt.title('Collision Ions-ions P=%i, J=%i' % (dims_i[0],dims_i[1]))
# plt.xlabel('(p,j)')
# plt.ylabel('(p,j)')
# plt.gca().invert_yaxis()
# plt.colorbar()

# plt.show()
A = np.zeros([36,36])
row = 0
column = 0
for i in range(66):
    
    for j in range(66):
        if np.floor(i/6)%2 == 0 and np.floor(j/6)%2 == 0:
            print(row,column)
            A[row,column] = matrix_ee[i,j]
            column+=1
            if column == 36:
                row+=1
                column=0
A[abs(A) <= 0.00] = np.nan
plt.figure()
plt.pcolor(A)
plt.title('submatrix_ee_even' )
plt.xlabel('(p,j)')
plt.ylabel('(p,j)')
plt.gca().invert_yaxis()
plt.colorbar()


B= np.zeros([30,30])
row = 0
column = 0
for i in range(66):
    
    for j in range(66):
        if np.floor(i/6)%2 == 1 and np.floor(j/6)%2 == 1:
            print(row,column)
            B[row,column] = matrix_ee[i,j]
            column+=1
            if column == 30:
                row+=1
                column=0

B[abs(B) <= 0.00] = np.nan
plt.figure()
plt.pcolor(B)
plt.title('submatrix_ee_odd' )
plt.xlabel('(p,j)')
plt.ylabel('(p,j)')
plt.gca().invert_yaxis()
plt.colorbar()


# # with h5.File('Matrix_ee_even.h5', 'w') as hf:
# #     hf.create_dataset('matrix_dataset', data=A)

# # with h5.File('Matrix_ee_odd.h5', 'w') as hf:
# #     hf.create_dataset('matrix_dataset', data=B)


# # dens_ee_even = abs(np.sum(A, axis=1))
# # plt.figure()
# # plt.plot(np.linspace(0,30,30),dens_ee_even[6:36])

# # plt.xlabel('Value')
# # plt.ylabel('Frequency')
# # plt.title('Histogram Plot')

# # Show the plot




# # dens_ee_odd = abs(np.sum(B, axis=1))

# # plt.plot(np.linspace(0,30,30),dens_ee_odd)

# # plt.xlabel('Value')
# # plt.ylabel('Frequency')
# # plt.title('Histogram Plot')



# # dens_ii = np.sum(correlation_matrix_ii, axis=0 )

U_even,S_even,VT_even = np.linalg.svd(A)
U_odd,S_odd,VT_odd = np.linalg.svd(B)
print(S_even)
print(S_odd)
# # print(np.linalg.norm(S_even[0:30] - S_odd)/np.linalg.norm(S_even[0:30]))

# # print(np.linalg.norm(dens_ee_even[6:36] - dens_ee_odd))

# # plt.figure()
# # plt.pcolor(A[6:36,6:36] - B)
# # plt.title('Collision Ions-ions P=%i, J=%i' % (dims_i[0],dims_i[1]))
# # plt.xlabel('(p,j)')
# # plt.ylabel('(p,j)')
# # plt.gca().invert_yaxis()
# # plt.colorbar()

# # # Show the plot
plt.show()

for i in range ( matrix_ee.shape[0]):
    magnitude = np.linalg.norm(abs(matrix_ee[i,:]))
    relevance = abs(matrix_ee[i,:])
    plt.plot(np.arange(0,matrix_ee.shape[0]) - i, relevance, label = '{}'.format(i))
    plt.title('matrix_ee')
    plt.legend()

plt.show()

for i in range (matrix_ii.shape[0]):
    magnitude = np.linalg.norm(abs(matrix_ii[i,:]))
    relevance = matrix_ii[i,:] / magnitude
    plt.plot(np.arange(0,matrix_ii.shape[0]) - i, relevance, label = '{}'.format(i))
    plt.title('matrix_ii')
    plt.legend()


plt.show()
S_even /= np.linalg.norm(S_even)
S_odd /= np.linalg.norm(S_odd)
plt.plot(np.arange(S_even.shape[0]),S_even)
plt.plot(np.arange(S_odd.shape[0]),S_odd)
plt.show()

for i in range (1,B.shape[0]):
    magnitude = np.linalg.norm(abs(B[i,:]))
    relevance = abs(B[i,:]) 
    plt.plot(np.arange(0,B.shape[0]) - i, relevance, label = '{}'.format(i))
    plt.title('even_submatrix')
    plt.legend()

plt.show()

errs = []
for k in range(66):
    trial1 = np.zeros(matrix_ee.shape)
    for i in range(matrix_ee.shape[0]):
        for j in range(matrix_ee.shape[1]):
            if ( abs(i-j) <k) :
                trial1[i,j] = matrix_ee[i,j]
                
    trial_err1 =0 
    iters = 100
    for _ in range(iters):
        coeff = np.random.rand(matrix_ee.shape[1])
        sol = matrix_ee@coeff
        sol_trial1 = trial1@coeff
        trial_err1 += np.linalg.norm(sol - sol_trial1)/np.linalg.norm(sol)/iters
    errs.append(trial_err1)

plt.plot(np.arange(66), errs)
plt.show()

# red = np.array([1,0,0])
# blue = np.array([0,0,1])

#f = lambda x: 1/np.sqrt(2*np.pi*270)*np.exp(-x**2/(2*270))
for i in range (1, matrix_ee.shape[0] -1):
    magnitude = np.linalg.norm(abs(matrix_ee[i,:]))
    relevance = abs(matrix_ee[i,:])
    if i%12 == 1:
        plt.plot(np.arange(0,matrix_ee.shape[0]) - i, relevance, label = '{}'.format(i))#, color = (1-i/65)*blue + i/65*(red))
        plt.title('matrix_ee')
        plt.legend()
    #plt.plot(np.linspace(-60,60,400), 175*f(np.linspace(-60,60,400)), 'o')
plt.show()

errs = []
for k in range(66):
    for l in range(66):
        trial1 = np.zeros(matrix_ee.shape)
        for i in range(matrix_ee.shape[0]):
            for j in range(matrix_ee.shape[1]):
                if ( (i-j) <k or (i-j)>-l) :
                    trial1[i,j] = matrix_ee[i,j]
                    
        trial_err1 =0 
        iters = 100
        for _ in range(iters):
            coeff = np.random.rand(matrix_ee.shape[1])
            sol = matrix_ee@coeff
            sol_trial1 = trial1@coeff
            trial_err1 += np.linalg.norm(sol - sol_trial1)/np.linalg.norm(sol)/iters
        errs.append(trial_err1)
plt.xlabel('Window size')
plt.ylabel('accuracy')
plt.plot(np.arange(66*66), errs)
plt.show()
print(np.argmin(errs), np.min(errs))