####### two moons simulation code refer to https://adapt-python.github.io/adapt/examples/Two_moons.html; 
####### we have modification such that the target space samples are i.i.d. samples (not directly rotate the samples in source space)
##### data simulation  ##########
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from algos import PDASMD_w_rounding
from sklearn.metrics import pairwise_distances
from scipy.stats import gaussian_kde
from sklearn.svm import SVC
#%%
def make_moons_da(n_samples=(150,150), rotation=50, noise=0.05, random_state1=10,random_state2=100):
    Xs, ys = make_moons(n_samples=n_samples,
                        noise=noise,
                        random_state=random_state1)
    Xs[:, 0] -= 0.5
    Xs[:, 1] -= 0.25
    
    Xt, yt = make_moons(n_samples=n_samples,
                        noise=noise,
                        random_state=random_state2)
    Xt[:, 0] -= 0.5
    Xt[:, 1] -= 0.25
    theta = np.radians(-rotation)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rot_matrix = np.array(
        ((cos_theta, -sin_theta),
         (sin_theta, cos_theta))
    )
    Xt = Xt.dot(rot_matrix)
    yt = yt
    return Xs, ys, Xt, yt

#%%
Xs, ys, Xt, yt = make_moons_da(rotation=90)

# x_min, y_min = np.min([Xs.min(0), Xt.min(0)], 0)
# x_max, y_max = np.max([Xs.max(0), Xt.max(0)], 0)
# x_grid, y_grid = np.meshgrid(np.linspace(x_min-0.1, x_max+0.1, 100),
#                              np.linspace(y_min-0.1, y_max+0.1, 100))
# X_grid = np.stack([x_grid.ravel(), y_grid.ravel()], -1)

fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
ax1.set_title("Source and target space")
ax1.scatter(Xs[ys==0, 0], Xs[ys==0, 1], label="source", c="red")
ax1.scatter(Xs[ys==1, 0], Xs[ys==1, 1], label="source", c="blue")
ax1.scatter(Xt[:, 0], Xt[:, 1], label="target",  c="black")
ax1.legend(loc="lower right")
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.tick_params(direction ='in')


plt.show()


#%% Cost function, then transportation plan

for degree in [10,20,30,40,50,70,90]:
    rotation = degree
    Xs, ys, Xt, yt = make_moons_da(rotation=rotation)

    C = pairwise_distances(Xs,Xt,metric = 'sqeuclidean')
    size = 300
    alpha = np.ones((size,1))/size
    beta = np.ones((size,1))/size
    trans_plan,_,_ = PDASMD_w_rounding(alpha = alpha, beta = beta, C = C,
            epsilon = 5e-3, Size = size, inner_iters = size, MD = True)
#fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
#ax1.set_title("Transported source space")
#ax1.scatter(hat_Xs[ys==0, 0], hat_Xs[ys==0, 1], label="transported source samples", c="red")
#ax1.scatter(hat_Xs[ys==1, 0], hat_Xs[ys==1, 1], label="transported source samples", c="blue")
#ax1.scatter(Xs[ys==0, 0], Xs[ys==0, 1], label="source", c="pink")
#ax1.scatter(Xs[ys==1, 0], Xs[ys==1, 1], label="source", c="green")
#ax1.scatter(Xt[:, 0], Xt[:, 1], label="target",  c="black")
#ax1.legend(loc="lower right")
#ax1.set_yticklabels([])
#ax1.set_xticklabels([])
#ax1.tick_params(direction ='in')
#plt.show()
#
    hat_Xs = size * (trans_plan @ Xt)
    
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(hat_Xs, ys)
    
    # create a mesh to plot decision boundary
    h = .02
    x_min, x_max = hat_Xs[:, 0].min() - 1, hat_Xs[:, 0].max() + 1
    y_min, y_max = hat_Xs[:, 1].min() - 1, hat_Xs[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = svclassifier.predict(np.c_[xx.ravel(), yy.ravel()])
    #Z = Z[:,1]
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    plt.figure(dpi=1000)
    plt.rcParams["figure.figsize"] = (5,5)
    
    plt.contour(xx, yy, Z, cmap='RdBu', alpha=0.8)
    #plt.pcolormesh(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8,vmin=Z.min(), vmax=Z.max())
    
    x1,y1 = hat_Xs[ys==0, 0], hat_Xs[ys==0, 1]
    xy1 = np.vstack([x1,y1])
    z1 = gaussian_kde(xy1)(np.c_[xx.ravel(), yy.ravel()].T)
    
    x2,y2 = hat_Xs[ys==1, 0], hat_Xs[ys==1, 1]
    xy2 = np.vstack([x2,y2])
    z2 = gaussian_kde(xy2)(np.c_[xx.ravel(), yy.ravel()].T)
    
    plt.scatter(xx.ravel(), yy.ravel(), c = z1/(max(z1) - min(z1)) - z2 /(max(z2)- min(z2)), s=5 ,cmap = 'RdBu')
    
    plt.scatter(Xs[ys==0, 0], Xs[ys==0, 1], label="source", c="tab:blue",s = 2)
    plt.scatter(Xs[ys==1, 0], Xs[ys==1, 1], label="source", c="tab:red",s = 2)
    
    plt.scatter(Xt[:, 0], Xt[:, 1], label="target",  c="black",s = 2)
    plt.legend(loc="lower right")
    plt.savefig('OT domain adaptation degree 90.pdf', format="pdf", bbox_inches = 'tight')

#%%
#For rotation 10, the average error from domain adaptation is 0.0
#For rotation 20, the average error from domain adaptation is 0.0
#For rotation 40, the average error from domain adaptation is 0.12304999999999996
#For rotation 50, the average error from domain adaptation is 0.17649999999999996
#For rotation 60, the average error from domain adaptation is 0.23215
#For rotation 90, the average error from domain adaptation is 0.4135

### make moons without specify randonstate
def make_moons_da2(n_samples=(150,150), rotation=50, noise=0.05):
    Xs, ys = make_moons(n_samples=n_samples,
                        noise=noise)
    Xs[:, 0] -= 0.5
    Xs[:, 1] -= 0.25
    
    Xt, yt = make_moons(n_samples=n_samples,
                        noise=noise)
    Xt[:, 0] -= 0.5
    Xt[:, 1] -= 0.25
    theta = np.radians(-rotation)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rot_matrix = np.array(
        ((cos_theta, -sin_theta),
         (sin_theta, cos_theta))
    )
    Xt = Xt.dot(rot_matrix)
    yt = yt
    return Xs, ys, Xt, yt


np.random.seed(1010)
#### calculate average error

for degree in [20,30,40,50,70,90]:
    rotation = degree
    
    #% Evaluate the error rate with repeated experiment
    errors = []
    for i in range(10):
        print('Running {}/10 experiment'.format(i+1))
        Xs, ys, Xt, yt = make_moons_da2(rotation=rotation)
    
        C = pairwise_distances(Xs,Xt,metric = 'sqeuclidean')
        size = 300
        alpha = np.ones((size,1))/size
        beta = np.ones((size,1))/size
        trans_plan,_,_ = PDASMD_w_rounding(alpha = alpha, beta = beta, C = C,
                epsilon = 5e-3, Size = size, inner_iters = size, MD = True)
    #
        hat_Xs = size * (trans_plan @ Xt)
        
        svclassifier = SVC(kernel='rbf')
        svclassifier.fit(hat_Xs, ys)
        
        X_test, y_test = make_moons(n_samples=(1000,1000), noise=0.05)
        X_test[:, 0] -= 0.5
        X_test[:, 1] -= 0.25
        theta = np.radians(-rotation)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        rot_matrix = np.array(
            ((cos_theta, -sin_theta),
             (sin_theta, cos_theta))
        )
        X_test = X_test.dot(rot_matrix)
        
        errors.append(sum(svclassifier.predict(X_test) != y_test)/len(y_test))
        
    print("For rotation {}, the average error from domain adaptation is {}".format(rotation,sum(errors)/len(errors)))
