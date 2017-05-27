def mult_synthesis_fuzzy(w_low, w_up, wc_low, wc_up):
    """
        Performs multiplicative synthesis on given interval weight matrix [[w11,...,wn1],...[w1m,...,wnm]]
        and criterion weight interval vector [wc1,...,wcm].
        """
    m = w_low.shape[0]
    n = w_low.shape[1]
    w_glob_low, w_glob_up = np.zeros(n), np.zeros(n)
    for i in range(n):
        c_low = -np.log(w_low[:, i])
        c_up = np.log(w_up[:, i])
        A_ub = np.vstack((-np.eye(m), np.eye(m)))
        b_ub = np.stack((wc_low, wc_up))
        A_eq = np.ones((1, m))
        b_eq = np.array([1])
        sol_low = linprog(c_low, A_ub, b_ub, A_eq, b_eq)
        w_glob_low[i] = np.exp(-sol_low.fun)
        sol_up = linprog(c_up, A_ub, b_ub, A_eq, b_eq)
        w_glob_up[i] = np.exp(sol_up.fun)
    return w_glob_low / np.linalg.norm(w_glob_low, ord=1), w_glob_up / np.linalg.norm(w_glob_up, ord=1)
