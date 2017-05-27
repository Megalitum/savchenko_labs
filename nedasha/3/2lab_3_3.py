
# coding: utf-8

# In[24]:

import numpy as np
from numpy import linalg


# In[25]:

# Get from lab2_1
def cr_metric(lamb, dim):
    """
    Eats matrix of pairwise comparisons m and produces CR metric of consistensy.
    """
    # calculate ci metric
    assert lamb >= dim, "Greatest eigenvalue must be greater than matrix dimension."
    ci = (lamb - dim)
    if dim > 1:
        ci /= float(dim - 1)
    # fill the MRCI table
    mrci = [0, 0, 0, 0.52, 0.89, 1.11, 1.25, 1.35, 1.40, 1.45, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59] 
    cr = 0 if ci == 0 else ci / mrci[dim]
    return cr

def em_cr(mat):
    """
    Eats matrix of pairwise comparisons m and produces pair (local weights, CR metric).
    """
    assert mat.shape[0] == mat.shape[1]
    values, vectors = linalg.eig(mat)
    max_index = np.argmax(abs(values))
    v = vectors[:, max_index]
    lamb = values[max_index]
    assert (v.imag < 1e-6).all(), "eigenvector has complex coordinates: %s" % v.imag
    assert lamb.imag < 1e-6, "eigenvalue has complex coordinates: %s" % lamb.imag
    v = v.real
    lamb = lamb.real
    if (v < 0).all():
        v = - v
    assert (v >= 0).all(), "eigenvector has negative coordinates: %s" % v
    return v / linalg.norm(v, ord=1), cr_metric(lamb, mat.shape[0])


# In[ ]:




# In[34]:

from_file = np.load('impc.npz')
a_low = from_file['a_low']
print 'a_low\n', a_low
a_up = from_file['a_up']
print '\na_up\n', a_up


# In[29]:

# geometric mean
A = np.sqrt(a_low * a_up)
print(A)


# In[39]:

weight, CR_metric = em_cr(A)
sorted_a = sorted(range(1, len(weight)+1), key= lambda i: weight[i-1], reverse=True)
print 'CR:\n', CR_metric
print 'Weight:\n', weight
print 'Sorted:\n', sorted_a


# In[ ]:




# In[ ]:



