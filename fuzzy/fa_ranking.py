import numpy as np
import sys

def fuzzy_advantage_rank(w_low, w_mid, w_up, j_s=0.2, j_e=0.2):
    mm = np.meshgrid(w_mid, w_mid)
    lu = np.meshgrid(w_low, w_up)
    vss = np.zeros_like(mm[0])
    vss[mm[1] >= mm[0]] = 1
    mask_mixed = (mm[1] < mm[0]) * (lu[1] >= lu[0])
    vals_mixed = (lu[1] - lu[0])[mask_mixed] / (lu[1] - mm[1] + mm[0] - lu[0])[mask_mixed]
    vss[mask_mixed] = vals_mixed
    print('Matrix V:\n',vss)
    vs = np.clip(vss - vss.T, 0, 1)
    ve = np.minimum(vss, vss.T)
    order_g = vs >= j_s
    order_eq = ve >= j_e
    print('">" relation:\n', order_g)
    print('"~" relation:\n', order_eq)
    return rank_by_g_order(order_g)

def rank_by_g_order(order):
    marked_indices = np.arange(order.shape[0])
    while (marked_indices.size > 0):
        order_slice = order[np.ix_(marked_indices, marked_indices)]
        greatest = np.where(np.sum(order_slice, axis=0) == 0)[0]
        yield marked_indices[greatest]
        marked_indices = np.delete(marked_indices, greatest)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        filename = 'f_weights.npz'
    else:
        filename = sys.argv[1]
    # load stored matrices
    with np.load(filename) as f:
        w_low = f['w_low']
        w_up = f['w_up']
        print('Vector succesfully loaded (%d)' % w_low.shape)
    ranking = fuzzy_advantage_rank(w_low, (w_low + w_up) / 2,  w_up)
    print('Ranking: ', ' '.join(str(r + 1) for r in ranking))
