import numpy as np

def GD(loss, K, x0, gamma=None, return_x=False):
    # x^{k+1} = x^k - \gamma * \nabla f(x^k)
    full_batch = np.arange(loss.n)
    f = np.zeros(K+1)
    f[0] = loss.func(x0, full_batch)
    x = [x0 for i in range(K+1)]

    if gamma is None:
        gamma = 1/loss.L
        name = r'GD, $\gamma=\frac{1}{L}$'
    else:
        name = r'GD, $\gamma={:.3f}'.format(gamma)+'$'

    for k in range(K):
        x[k+1] = x[k] - gamma * loss.grad(x[k], full_batch)
        f[k+1] = loss.func(x[k+1], full_batch)

    if return_x:
        return name, f, x
    else:
        return name, f

def unifiedSAM(loss, trials, record_f, x0, gamma, rho, lambd, decreasing=False, importance=False, bs=1):
    # Unified SAM
    # x^{k+1} = x^k - \gamma * \nabla f_i(x^k + \rho * (1-\lambda + \lambda/|\nabla f_i(x^t)|) * \nabla f_i(x^k))
    EPS = 1e-7
    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    K = record_f[-1]

    for trial in range(trials):
        x = [x0 for i in range(K+1)]
        counter = 1

        for k in range(K):
            if importance:
                probs = loss.L_i/np.sum(loss.L_i)
                i_k = np.random.choice(a=range(loss.n), size=bs, p=probs)
            else:
                i_k = np.random.choice(a=range(loss.n), size=bs)
            grad_norm = np.linalg.norm(loss.grad(x[k], i_k))

            if decreasing:
                rho_k = min(rho, 1/(2*k+1))
                gamma_k = min(gamma, (2*k+1)/((k+1)**2*loss.mu))
            else:
                rho_k = rho
                gamma_k = gamma
            
            extr = x[k] + rho_k * (1 - lambd + lambd/(grad_norm+EPS)) * loss.grad(x[k], i_k)
            x[k+1] = x[k] - gamma_k * loss.grad(extr, i_k)

            if k+1 in record_f:
                f[counter, trial] = loss.func(x[k+1], full_batch)
                counter += 1

    name = r'SAM, $\lambda='+str(lambd)+r', $\rho='+"{:.3f}".format(rho)+', $\gamma='+"{:.3f}".format(gamma)+r'$'
    if decreasing:
        name += ' (Decreasing)'
    if importance:
        name += ' (Importance)'
    
    return name, f

def unifiedSAM_det(loss, K, x0, gamma, rho, lambd):
    full_batch = np.arange(loss.n)
    f = np.zeros(K+1)
    f[0] = loss.func(x0, full_batch)
    x = [x0 for i in range(K+1)]

    for k in range(K):
        grad_norm = np.linalg.norm(loss.grad(x[k], full_batch))
        extr = x[k] + rho * (1 - lambd + lambd/(grad_norm)) * loss.grad(x[k], full_batch)
        x[k+1] = x[k] - gamma * loss.grad(extr, full_batch)
        f[k+1] = loss.func(x[k+1], full_batch)

    name = r'Det SAM, $\lambda='+str(lambd)+r', $\rho='+"{:.3f}".format(rho)+', $\gamma='+"{:.3f}".format(gamma)+r'$'
    return name, f

def SAMDec(loss, trials, record_f, x0, PL=False, same_sample=True, bs=1):
    # Same/Different sample SAM
    # x^{k+1} = x^k - \gamma * \nabla f_i(x^k + \rho * \nabla f_j(x^k))
    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    K = record_f[-1]

    for trial in range(trials):
        x = [x0 for i in range(K+1)]
        counter = 1

        for k in range(K):
            i_k = np.random.choice(range(loss.n), bs)
            j_k = np.random.choice(range(loss.n), bs)
            if same_sample:
                j_k = i_k
            
            gamma = (1/loss.L)*K**(-1/2)
            rho = (1/loss.L)*K**(-1/4)
            if PL:
                gamma = min((8*k+4)/(3*(k+1)**2*loss.mu), 1/(2*loss.L))
                rho = np.sqrt(gamma/loss.L)
            
            extr = x[k] + rho * loss.grad(x[k], j_k)
            x[k+1] = x[k] - gamma * loss.grad(extr, i_k)

            if k+1 in record_f:
                f[counter, trial] = loss.func(x[k+1], full_batch)
                counter += 1

    if PL:
        name = r'SAM Decreasing PL, $\gamma_t=min{\frac{8t+4}{3(t+1)^2\mu}, \frac{1}{2L}}$, $\rho_t=\sqrt{\gamma_t/L}$'
    else:
        name = r'SAM Decreasing, $\gamma_t=\frac{1}{T^{1/2}L}$, $\rho_t=\frac{1}{T^{1/4}L}$'
    return name, f


def decSGD(loss, trials, record_f, x0, gamma, bs=1):
    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    K = record_f[-1]
    
    for trial in range(trials):
        x = [x0 for i in range(K+1)]
        counter = 1

        for k in range(K):
            i_k = np.random.choice(range(loss.n), bs)
            gamma_k = min(gamma, (2*k+1)/((k+1)**2*loss.mu))
            x[k+1] = x[k] - gamma_k * loss.grad(x[k], i_k)

            if k+1 in record_f:
                f[counter, trial] = loss.func(x[k+1], full_batch)
                counter += 1

    name = r'DecSGD, $\gamma='+"{:.3f}".format(gamma)+r'$'
    return name, f

def SGD(loss, trials, record_f, x0, gamma, bs=1):
    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    K = record_f[-1]

    for trial in range(trials):
        x = [x0 for i in range(K+1)]
        counter = 1

        for k in range(K):
            i_k = np.random.choice(range(loss.n), bs)
            x[k+1] = x[k] - gamma * loss.grad(x[k], i_k)

            if k+1 in record_f:
                f[counter, trial] = loss.func(x[k+1], full_batch)
                counter += 1

    name = r'SGD, $\gamma='+"{:.3f}".format(gamma)+r'$'
    return name, f
