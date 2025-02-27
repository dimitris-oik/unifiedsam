import numpy as np
from scipy.stats import ortho_group

"""
Loss function should have
- func: return f(x)
- grad: return \nabla f(x)
- L: smoothness
- L_i, L_max, L_avg
- mu: strong convexity
- kappa: L/mu
- regularization

- prox
"""

# helper functions
def sq_matrix_with_given_cond(n, cond):
    # result is always symmetric positive definite
    log_cond = np.log(cond)
    exp_vec = np.arange(-log_cond/4., log_cond * (n + 1)/(4 * (n - 1)), log_cond/(2.*(n-1)))
    exp_vec = exp_vec[:n]
    s = np.exp(exp_vec)
    S = np.diag(s)
    U, _ = np.linalg.qr((np.random.rand(n, n) - 5.) * 200)
    V, _ = np.linalg.qr((np.random.rand(n, n) - 5.) * 200)
    P = U.dot(S).dot(V.T)
    P = P.dot(P.T)

    return P

def matrix_with_given_cond(n, d, cond, symmetric=False):
    assert d >= 2
    assert n >= d
    P = ortho_group.rvs(dim=n)
    if symmetric:
        Q = P.T
    else:
        Q = ortho_group.rvs(dim=d)
    D = np.zeros((n, d))
    
    t = np.sqrt(cond)
    u = np.random.uniform(low=-1, high=1, size=d-2)
    np.insert(u, 0, -1)
    np.append(u, 1)
    np.fill_diagonal(D, np.float_power(t, u))
    
    A = P@D@Q
    return A


class LogisticRegression():
    def __init__(self, n, d, lambd=0.0, A=None, y=None, cond=None):
        self.name = "Logistic Regression"
        self.save_name = "logreg"
        self.n = n
        self.d = d
        assert self.d >= 2
        assert self.n >= self.d
        self.lambd = lambd
        self.A_cond = cond

        if A is not None:
            self.A = A
        else: # A == None
            if cond is not None and n == d: # cond = number, n == d
                self.A = sq_matrix_with_given_cond(n, cond)
            elif cond is not None and n != d: # cond = number, n != d
                self.A = matrix_with_given_cond(n, d, cond)
            else: # cond == None
                self.A = np.random.randn(n, d)

        if y is not None:
            self.y = y
        else: # y == None
            self.y = np.random.choice([-1, 1], size=self.n)

        self.L_i = 0.25*np.array([np.linalg.eig(np.outer(self.A[i], self.A[i]))[0].real.max() for i in range(self.n)]) + self.lambd
        self.L_max = self.L_i.max()
        self.L_avg = self.L_i.mean()
        self.L = np.linalg.eig(self.A.T@self.A)[0].real.max()/self.n + self.lambd
        self.mu = lambd
        if lambd > 0:
            self.kappa = self.L/self.mu
        else:
            self.kappa = None
    
    def func(self, x, ind):
        return np.sum(np.log(1+np.exp(-np.dot(self.A[ind], x) * self.y[ind])))/(2*ind.shape[0]) + (self.lambd/2)*np.linalg.norm(x)
    
    def grad(self, x, ind):
        batch_size = ind.shape[0]
        num = -self.y[ind]
        den = (1 + np.exp(np.dot(self.A[ind], x) * self.y[ind]))
        c = num/den
        c2 = np.dot(c.T, self.A[ind])
        final_grad = (c2)/(2*batch_size) + self.lambd * x
        return final_grad


class RidgeRegression():
    def __init__(self, n, d, lambd=0.0, A=None, b=None, cond=None, consistent=False):
        self.name = "Ridge Regression"
        self.save_name = "ridgereg"
        self.n = n
        self.d = d
        assert self.d >= 2
        assert self.n >= self.d
        self.lambd = lambd
        self.A_cond = cond

        if A is not None:
            self.A = A
        else: # A == None
            if cond is not None and n == d: # cond = number, n == d
                self.A = sq_matrix_with_given_cond(n, cond)
            elif cond is not None and n != d: # cond = number, n != d
                self.A = matrix_with_given_cond(n, d, cond)
            else: # cond == None
                self.A = np.random.randn(n, d)
        
        if consistent:
            self.f_star = 0.0
            self.x_star = np.random.randn(self.d)
            self.b = self.A@self.x_star
        else:
            self.f_star = None
            self.x_star = None
            if b is not None:
                self.b = b
            else: # b == None
                self.b = np.random.randn(self.n)

        self.L_i = np.array([np.linalg.eig(np.outer(self.A[i], self.A[i]))[0].real.max() for i in range(self.n)]) + self.lambd
        self.L_max = self.L_i.max()
        self.L_avg = self.L_i.mean()
        self.L = np.linalg.eig(self.A.T@self.A)[0].real.max()/self.n + self.lambd
        self.mu = np.linalg.eig(self.A.T@self.A)[0].real.min()/self.n + self.lambd
        self.kappa = self.L/self.mu

    def func(self, x, batch):
        return np.linalg.norm(self.A[batch]@x-self.b[batch])**2/(2*batch.shape[0]) + (self.lambd/2)*np.linalg.norm(x)**2
    
    def grad(self, x, batch):
        return (self.A[batch].T@(self.A[batch]@x-self.b[batch]))/batch.shape[0] + self.lambd*x


class RidgeRegression_():
    def __init__(self, n, d, lambd, mu, L, b=None, consistent=False):
        self.name = "Ridge Regression_"
        self.save_name = "ridgereg_"
        self.n = n
        self.d = d
        assert self.d >= 2
        assert self.n >= self.d
        self.lambd = lambd
        self.mu = mu
        self.L = L

        eigs = np.concatenate((np.array([mu, L]), np.random.uniform(low=mu, high=2*mu, size=d-2)))
        DA = np.diag(eigs)
        QA = ortho_group.rvs(d)
        self.A = QA@DA@QA.T
        
        if consistent:
            self.f_star = 0.0
            self.x_star = np.random.randn(self.d)
            self.b = self.A@self.x_star
        else:
            self.f_star = None
            self.x_star = None
            if b is not None:
                self.b = b
            else: # b == None
                self.b = np.random.randn(self.n)

        self.L_i = np.array([np.linalg.eig(np.outer(self.A[i], self.A[i]))[0].real.max() for i in range(self.n)]) + self.lambd
        self.L_max = self.L_i.max()
        self.L_avg = self.L_i.mean()
        self.L = np.linalg.eig(self.A.T@self.A)[0].real.max()/self.n + self.lambd
        self.mu = np.linalg.eig(self.A.T@self.A)[0].real.min()/self.n + self.lambd
        self.kappa = self.L/self.mu

    def func(self, x, batch):
        return np.linalg.norm(self.A[batch]@x-self.b[batch])**2/(2*batch.shape[0]) + (self.lambd/2)*np.linalg.norm(x)**2
    
    def grad(self, x, batch):
        return (self.A[batch].T@(self.A[batch]@x-self.b[batch]))/batch.shape[0] + self.lambd*x


class LeastSquares():
    # Deterministic equivalent of Ridge Regression
    def __init__(self, n, d, lambd=0.0, A=None, b=None, cond=None, consistent=False):
        self.name = "Least Squares"
        self.save_name = "leastsq"
        self.n = n
        self.d = d
        assert self.d >= 2
        assert self.n >= self.d
        self.lambd = lambd
        self.A_cond = cond

        if A is not None:
            self.A = A
        else: # A == None
            if cond is not None and n == d: # cond = number, n == d
                self.A = sq_matrix_with_given_cond(n, cond)
            elif cond is not None and n != d: # cond = number, n != d
                self.A = matrix_with_given_cond(n, d, cond)
            else: # cond == None
                self.A = np.random.randn(n, d)
        
        if consistent:
            self.f_star = 0.0
            self.x_star = np.random.randn(self.d)
            self.b = self.A@self.x_star
        else:
            self.f_star = None
            self.x_star = None
            if b is not None:
                self.b = b
            else: # b == None
                self.b = np.random.randn(self.n)

        eigs = np.linalg.eig(self.A.T@self.A)[0].real
        self.L = eigs.max() + self.lambd
        self.mu = eigs.min() + self.lambd
        self.kappa = self.L/self.mu

    def func(self, x, batch):
        return np.linalg.norm(self.A@x-self.b)**2/2 + (self.lambd/2)*np.linalg.norm(x)**2
    
    def grad(self, x, batch):
        return self.A.T@(self.A@x-self.b) + self.lambd*x
