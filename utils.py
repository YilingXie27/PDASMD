import numpy as np
from scipy.special import softmax
import itertools
from scipy.spatial.distance import cdist

def fullgrad(v, eta, C, alpha, beta, Size):### full gradient of semi-dual objective
    full=-Size*(alpha*(beta.T-softmax( (v.T-C-eta) /eta, axis=1)))
    return full


def batchgrad(v, eta, C, alpha, beta,idxes, Size): ### batch gradient of semi-dual objective, note that the stochastic gradient is a special case
    batch=-Size*(alpha[idxes]*(beta.T-softmax( (v.T-C[idxes]-eta) /eta, axis=1)))
    return batch

def primal_semidual(semi, alpha, beta, C, Size, eta):
     semi = np.reshape(semi,(Size,-1))
     p = (softmax( (semi.T-C-eta) /eta, axis=1)*alpha).T.reshape(-1,1)
     return p
 
def roundto(F, alpha, beta): #round solution to feasible area of OT
    min_vec = np.vectorize(min)
    x = min_vec(alpha / F.sum(axis = 1).reshape(-1, 1),1)
    X = np.diag(x.T[0])
    F1 = np.dot(X, F)
    y = min_vec(beta / F1.sum(axis = 0).reshape(-1, 1),1)
    Y = np.diag(y.T[0])
    F11 = np.dot(F1, Y)
    erra = alpha - F11.sum(axis = 1).reshape(-1,1)
    errb = beta - F11.sum(axis = 0).reshape(-1,1)
    G = F11 + np.dot(erra, errb.T) / np.linalg.norm(erra, ord = 1)
    return G

def phi_(h, gamma, C, Size, conalpha, conbeta): ### dual function 
    one = np.ones(Size, np.float64)
    A = (-C/gamma + np.outer(h[:Size], one) + np.outer(one, h[Size:]))
    a = A.max()
    A-=a
    s = a+np.log(np.exp(A).sum())
    return gamma*(-h[:Size].dot(conalpha) - h[Size:].dot(conbeta) + s)

def f_(gamma, h, C, Size): ### primal function
    y = (h.reshape(-1)).copy()
    y[h.reshape(-1) == 0.] = 1.
    y = y.reshape(Size, -1)
    return (C * h).sum() + gamma * (h * np.log(y)).sum()

#### function for generating a m * m image
def synthetic_img_input(m, fraction_fg = 0.2, seed = 123):
    np.random.seed(seed)
    fg_max_intensity = 3;
    bg_max_intensity = 1;

    IMG = bg_max_intensity * np.random.uniform(0, 1, (m,m))

    length_fg_side = int(np.floor(m * np.sqrt(fraction_fg)))
    img_i = np.random.choice(m - length_fg_side)
    img_j = np.random.choice(m - length_fg_side)
    
    for i in range(img_i,img_i + length_fg_side):
        for j in range(img_j,img_j+ length_fg_side):
            IMG[i, j] = fg_max_intensity * np.random.uniform(0,1)
    
    IMG/=IMG.sum()
    IMG = IMG.reshape(-1,1)
    
    return IMG 

#### Input dimension m, output the cost matrices between two m * m images as measured by the l1 distance between pixels
def dist_img_m(m):
    C = list(itertools.product(range(m),range(m)))
    C = cdist(C, C, 'minkowski', p=1)
    C /= np.max(C)
    return C