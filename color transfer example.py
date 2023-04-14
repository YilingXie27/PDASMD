#Some implementations refer to https://github.com/PythonOT/POT/tree/master/ot
import numpy as np
from ot.utils import unif, dist, cost_normalization, label_normalization#, kernel, laplacian, dots
from ot.utils import check_params, BaseEstimator
#from ot.optim import cg
#from ot.optim import gcg
from algos import PDASMD
from matplotlib import pyplot as plt

#%% Define useful functions and classes
def im2mat(img):
    """Converts an image to matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))


def mat2im(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)


def minmax(img):
    return np.clip(img, 0, 1)

def distribution_estimation_uniform(X):
    return unif(X.shape[0]).reshape(-1,1)


class BaseTransport(BaseEstimator):
    def fit(self, Xs=None, ys=None, Xt=None, yt=None):
        # check the necessary inputs parameters are here
        if check_params(Xs=Xs, Xt=Xt):

            # pairwise distance
            self.cost_ = dist(Xs, Xt, metric=self.metric)
            self.cost_ = cost_normalization(self.cost_, self.norm)

            if (ys is not None) and (yt is not None):

                if self.limit_max != np.infty:
                    self.limit_max = self.limit_max * np.max(self.cost_)

                # assumes labeled source samples occupy the first rows
                # and labeled target samples occupy the first columns
                classes = [c for c in np.unique(ys) if c != -1]
                for c in classes:
                    idx_s = np.where((ys != c) & (ys != -1))
                    idx_t = np.where(yt == c)

                    # all the coefficients corresponding to a source sample
                    # and a target sample :
                    # with different labels get a infinite
                    for j in idx_t[0]:
                        self.cost_[idx_s[0], j] = self.limit_max

            # distribution estimation
            self.mu_s = self.distribution_estimation(Xs)
            self.mu_t = self.distribution_estimation(Xt)

            # store arrays of samples
            self.xs_ = Xs
            self.xt_ = Xt

        return self

    def fit_transform(self, Xs=None, ys=None, Xt=None, yt=None):
        return self.fit(Xs, ys, Xt, yt).transform(Xs, ys, Xt, yt)

    def transform(self, Xs=None, ys=None, Xt=None, yt=None, batch_size=128):

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs):

            if np.array_equal(self.xs_, Xs):

                # perform standard barycentric mapping
                transp = self.coupling_ / np.sum(self.coupling_, 1)[:, None]

                # set nans to 0
                transp[~ np.isfinite(transp)] = 0

                # compute transported samples
                transp_Xs = np.dot(transp, self.xt_)
            else:
                # perform out of sample mapping
                indices = np.arange(Xs.shape[0])
                batch_ind = [
                    indices[i:i + batch_size]
                    for i in range(0, len(indices), batch_size)]

                transp_Xs = []
                for bi in batch_ind:
                    # get the nearest neighbor in the source domain
                    D0 = dist(Xs[bi], self.xs_)
                    idx = np.argmin(D0, axis=1)

                    # transport the source samples
                    transp = self.coupling_ / np.sum(
                        self.coupling_, 1)[:, None]
                    transp[~ np.isfinite(transp)] = 0
                    transp_Xs_ = np.dot(transp, self.xt_)

                    # define the transported points
                    transp_Xs_ = transp_Xs_[idx, :] + Xs[bi] - self.xs_[idx, :]

                    transp_Xs.append(transp_Xs_)

                transp_Xs = np.concatenate(transp_Xs, axis=0)

            return transp_Xs

    def transform_labels(self, ys=None):
        # check the necessary inputs parameters are here
        if check_params(ys=ys):

            ysTemp = label_normalization(np.copy(ys))
            classes = np.unique(ysTemp)
            n = len(classes)
            D1 = np.zeros((n, len(ysTemp)))

            # perform label propagation
            transp = self.coupling_ / np.sum(self.coupling_, 0, keepdims=True)

            # set nans to 0
            transp[~ np.isfinite(transp)] = 0

            for c in classes:
                D1[int(c), ysTemp == c] = 1

            # compute propagated labels
            transp_ys = np.dot(D1, transp)

            return transp_ys.T

    def inverse_transform(self, Xs=None, ys=None, Xt=None, yt=None,
                          batch_size=128):
        # check the necessary inputs parameters are here
        if check_params(Xt=Xt):

            if np.array_equal(self.xt_, Xt):

                # perform standard barycentric mapping
                transp_ = self.coupling_.T / np.sum(self.coupling_, 0)[:, None]

                # set nans to 0
                transp_[~ np.isfinite(transp_)] = 0

                # compute transported samples
                transp_Xt = np.dot(transp_, self.xs_)
            else:
                # perform out of sample mapping
                indices = np.arange(Xt.shape[0])
                batch_ind = [
                    indices[i:i + batch_size]
                    for i in range(0, len(indices), batch_size)]

                transp_Xt = []
                for bi in batch_ind:
                    D0 = dist(Xt[bi], self.xt_)
                    idx = np.argmin(D0, axis=1)

                    # transport the target samples
                    transp_ = self.coupling_.T / np.sum(
                        self.coupling_, 0)[:, None]
                    transp_[~ np.isfinite(transp_)] = 0
                    transp_Xt_ = np.dot(transp_, self.xs_)

                    # define the transported points
                    transp_Xt_ = transp_Xt_[idx, :] + Xt[bi] - self.xt_[idx, :]

                    transp_Xt.append(transp_Xt_)

                transp_Xt = np.concatenate(transp_Xt, axis=0)

            return transp_Xt

    def inverse_transform_labels(self, yt=None):
        # check the necessary inputs parameters are here
        if check_params(yt=yt):

            ytTemp = label_normalization(np.copy(yt))
            classes = np.unique(ytTemp)
            n = len(classes)
            D1 = np.zeros((n, len(ytTemp)))

            # perform label propagation
            transp = self.coupling_ / np.sum(self.coupling_, 1)[:, None]

            # set nans to 0
            transp[~ np.isfinite(transp)] = 0

            for c in classes:
                D1[int(c), ytTemp == c] = 1

            # compute propagated samples
            transp_ys = np.dot(D1, transp.T)

            return transp_ys.T


class PDASMDTransport(BaseTransport):
    def __init__(self, reg_e=1e-2, max_iter=1000,
                 tol=1e-2, verbose=False, log=False,
                 metric="sqeuclidean", norm=None,
                 distribution_estimation=distribution_estimation_uniform,
                 out_of_sample_map='ferradans', limit_max=np.infty):
        self.reg_e = reg_e
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.log = log
        self.metric = metric
        self.norm = norm
        self.limit_max = limit_max
        self.distribution_estimation = distribution_estimation
        self.out_of_sample_map = out_of_sample_map

    def fit(self, Xs=None, ys=None, Xt=None, yt=None):
        super(PDASMDTransport,self).fit(Xs, ys, Xt, yt)

        # coupling estimation
        self.size = len(self.mu_s)
        returned_,_,_ = PDASMD(alpha = self.mu_s, beta = self.mu_t, inner_iters = self.size,
                               acc = self.tol, penalty=self.reg_e, C = self.cost_)
        # deal with the value of log
        if self.log:
            self.coupling_, self.log_ = returned_
        else:
            self.coupling_ = returned_
            self.log_ = dict()

        return self

#%%% Color transfer example
rng = np.random.RandomState(42)

# Loading images
I1 = plt.imread('chicken.jpg').astype(np.float64) / 256
I2 = plt.imread('turtle.jpg').astype(np.float64) / 256

X1 = im2mat(I1)
X2 = im2mat(I2)

# training samples
nb = 500
idx1 = rng.randint(X1.shape[0], size=(nb,))
idx2 = rng.randint(X2.shape[0], size=(nb,))

Xs = X1[idx1, :]
Xt = X2[idx2, :]

alpha=distribution_estimation_uniform(Xs)
beta=distribution_estimation_uniform(Xt)

C = dist(Xs)


ot_TestPDASMD1 = PDASMDTransport()
ot_TestPDASMD1.fit(Xs=Xs,Xt=Xt)


transp_Xs_PDASMD1 = ot_TestPDASMD1.transform(Xs=X1)
transp_Xt_PDASMD1 = ot_TestPDASMD1.inverse_transform(Xt=X2)


I1tePDASMD1 = minmax(mat2im(transp_Xs_PDASMD1, I1.shape))
I2tePDASMD1 = minmax(mat2im(transp_Xt_PDASMD1, I2.shape))
#%%
plt.subplot(1, 3, 1)
plt.imshow(I1)
plt.axis('off')
plt.title('Source')

plt.subplot(1, 3, 2)
plt.imshow(I2)
plt.axis('off')
plt.title('Target')

plt.subplot(1, 3,3)
plt.imshow(I1tePDASMD1)
plt.axis('off')
plt.title('Transferred')

plt.savefig('PDASGD color transfer.jpg')



