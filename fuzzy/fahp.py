import numpy as np
import sys

def fuzzy_advantage(a_low, a_mid, a_up):
    a = np.stack((a_low, a_mid, a_up))
    rsi = a.sum(axis=2)
    si = rsi / rsi.sum(axis=1)[::-1, None]
    mm = np.meshgrid(si[1], si[1]) # mm[1] ~ m_i, mm[0] ~ m_j 
    lu = np.meshgrid(si[0], si[2]) # lu[1] ~ u_i, lu[0] ~ l_j
    vss = np.zeros_like(a_low)
    vss[mm[1] >= mm[0]] = 1
    mask_mixed = (mm[1] < mm[0]) * (lu[1] >= lu[0])
    vals_mixed = (lu[1] - lu[0]) / (lu[1] - mm[1] + mm[0] - lu[0])
    vss[mask_mixed] = vals_mixed[mask_mixed]
    print('Matrix S(l,m,u):\n', si.T)
    print('Matrix VS:\n',vss)
    weights = np.amin(vss, axis=1)
    # calculated weights
    return weights / np.linalg.norm(weights, 1) 

if __name__ == '__main__':
    if len(sys.argv) < 2:
        filename = 'impc.npz'
    else:
        filename = sys.argv[1]
    # load stored matrices
    with np.load(filename) as f:
        a_low = f['a_low']
        a_up = f['a_up']
        print('Matrices succesfully loaded (%dx%d)' % a_low.shape)
    weights = fuzzy_advantage(a_low, (a_low + a_up) / 2,  a_up)
    print('Weights: ', ' '.join(str(w) for w in weights))
