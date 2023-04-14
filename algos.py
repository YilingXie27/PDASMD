###Some implementations refer to https://github.com/PythonOT/POT/tree/master/ot, https://github.com/JasonAltschuler/OptimalTransportNIPS17 and https://github.com/nazya/AAM. 

import numpy as np
import time
from utils import fullgrad,batchgrad,primal_semidual,roundto, phi_, f_
import scipy
import warnings

#%% PDASMD
def PDASMD(alpha, beta, inner_iters, acc, penalty, C, batchsize = 1, seed = 123, MD = True): 
    ###  MD True corresponding to l_\infty norm, MD False corresponding to l_2 norm
    ### batchsize default to 1, one can set it up to \sqrt{n}. remember to adjust inner_iters accordingly
      time0 = time.time()
      Size = C.shape[0]
      np.random.seed(seed) 
      tau2 = 1/2/batchsize
      s=0 
      y_store = np.zeros((Size, 1),np.float64)
      z_temp = np.zeros((Size, 1),np.float64)
      v_temp = np.zeros((Size, 1),np.float64) 
      vtilde = np.zeros((Size, 1),np.float64)
      x_s = np.zeros((Size, Size),np.float64)
      C_temp = 0
      D_temp = np.zeros((Size * Size, 1),np.float64)
      Lip = (1 / penalty)
      
      flag = 0 
      Store_Error = []
      Store_Flag = []
      Store_Time = []
      Store_value = []
      acc_temp = abs(x_s.sum(axis=1).reshape(-1,1) - alpha).sum() + abs(x_s.sum(axis=0).reshape(-1, 1) - beta).sum()
      valueotinitial = np.dot(C.reshape(1,-1),x_s.reshape(-1,1))[0][0]
      Store_Error.append(acc_temp)
      Store_Flag.append(0)
      Store_Time.append(0)
      Store_value.append(valueotinitial)
      timecost = 0
      error=100
      
      while error >= acc: 

          tau1 = 2 / (s + 4)
          flag = flag + 2
          
          alphas = 1 / (9 * tau1 * Lip)
          flag =  flag + 3
          
          recorded_gradients = fullgrad(v = vtilde, eta = penalty, C = C, alpha = alpha, beta = beta, Size = Size)
          flag = flag + 9 * Size * Size -  Size
          
          full_gradient = recorded_gradients.mean(axis=0).reshape(-1,1) 
          flag = flag + Size * Size
          
          store_y=[]
          for _ in range(inner_iters):      
                j = np.random.choice(Size,batchsize,p = alpha.reshape(-1))
                v_temp = tau1 * z_temp + tau2 * vtilde + (1 - tau1 - tau2) * y_store
                flag = flag + 5 * Size + 2
                
                grad_temp = full_gradient + ( 
                    Size * alpha[j] * 
                    (batchgrad(v = v_temp, eta = penalty, C = C, alpha = alpha, beta = beta, idxes = j, Size = Size)
                                                - recorded_gradients[j]) ).mean(axis=0).reshape(-1,1) 
                flag = flag + 8 * Size * batchsize 
                flag = flag + Size
                
                z_temp = z_temp - alphas * grad_temp 
                flag = flag + 2 * Size
                
                if MD:
                    y_store = v_temp - np.linalg.norm(grad_temp, ord=1) / (9 * Lip) * np.sign(grad_temp) 
                    flag = flag + 4 * Size + 1
                else:
                    y_store = v_temp - grad_temp / (9 * Lip)
                    flag = flag + 2 * Size + 1
                
                store_y=np.append(store_y,y_store) 
          store_y = np.reshape(store_y,(-1,Size)) 
          vtilde = store_y.mean(axis = 0).reshape(-1,1)
          flag = flag + inner_iters
          
          C_temp = C_temp + 1 / tau1
          flag = flag + 2
          
          t = np.random.choice(inner_iters)
          random_y = store_y[t,:].reshape(-1,1)
          D_temp = D_temp + (1 / tau1) * primal_semidual(random_y, alpha = alpha, beta = beta, C =C, Size = Size, eta = penalty) 
          flag = flag + 7 * Size * Size - Size
          flag = flag + Size * Size + Size * Size +1
          
          x_s = (D_temp/C_temp).reshape(Size, Size).T #convert to matrix
          flag = flag + Size * Size
          
          s = s + 1 
          
          
          
          error = abs(x_s.sum(axis=1).reshape(-1,1) - alpha).sum() + abs(x_s.sum(axis=0).reshape(-1, 1) - beta).sum()
          Store_Error.append(error)
          Store_Flag.append(flag)

          
          valueot = np.dot(C.reshape(1,-1),x_s.reshape(-1,1))[0][0]
          Store_value.append(valueot)
          
          
          timecost = time.time() - time0
          Store_Time.append(timecost)
          
          
      return x_s, flag, timecost
  
def PDASMD_w_rounding(alpha, beta, C, epsilon, Size, inner_iters, batchsize = 1, MD = True):
    Size = C.shape[0]
    penalty = epsilon / (4 * np.log(Size))
    epsilonprime = epsilon / 8 * abs(C).max()
    alphatilde = (1 - epsilonprime / 8) * alpha + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    betatilde = (1 - epsilonprime / 8) * beta + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    Xtilde, computations,time = PDASMD(alpha = alphatilde, beta = betatilde , inner_iters = inner_iters,
           acc = epsilonprime / 2, penalty = penalty, C = C, batchsize = batchsize, MD = MD)
    Xhat = roundto(Xtilde, alpha = alpha, beta = beta)   
    return Xhat, computations, time


#%% APDAGD
def APDAGD(acc,conalpha,conbeta,C,gamma,Size):
    time0=time.time()
    one = np.ones(Size, np.float64)
    L = 1.
    betta = 0.
    primal_var = 0.*np.ones([Size, Size], np.float64)
    z = np.zeros(2*Size, np.float64)
    z_new = np.zeros(2*Size, np.float64)
    eta = np.zeros(2*Size, np.float64)
    eta_new  = np.zeros(2*Size, np.float64)
    grad_phi_new = np.zeros(2*Size, np.float64)    
    f = lambda h: phi_(h, gamma=gamma, C=C, Size=Size, conalpha=conalpha, conbeta=conbeta)
    
    
    k=0
    oper_num=0
    
    
    K=-C/gamma
    oper_num = oper_num + Size * Size
    
    Store_Acc=[]
    Store_Oper=[]
    Store_Time=[]
    Store_value = []
    valueotinitial = np.dot(C.reshape(1,-1),primal_var.reshape(-1,1))[0][0]
    Store_value.append(valueotinitial)
    acc_temp = abs(np.squeeze(primal_var.sum(axis=1).reshape(-1,1)) - conalpha).sum() + abs(np.squeeze(primal_var.sum(axis=0).reshape(-1, 1)) - conbeta).sum()
    Store_Acc.append(acc_temp)
    Store_Oper.append(0)
    Store_Time.append(0)

    while True:
        L = L / 2
        oper_num = oper_num + 1
        while True:
            alpha_new = (1 + np.sqrt(4*L*betta + 1)) / 2 / L
            oper_num = oper_num + 7
            
            betta_new = betta + alpha_new   
            oper_num = oper_num + 1
            
            tau = alpha_new / betta_new
            oper_num = oper_num +1
            
            lamu_new = tau * z + (1 - tau) * eta   
            oper_num = oper_num + 2 * Size * 2 + 1                                 
            
            logB = (K + np.outer(lamu_new[:Size], one) + np.outer(one, lamu_new[Size:]))
            oper_num = oper_num + 4 * Size * Size
            
            max_logB =logB.max()
            logB_stable = logB - max_logB
            oper_num = oper_num + Size * Size
            
            B_stable = np.exp(logB_stable)
            oper_num = oper_num + Size * Size
            
            u_hat_stable, v_hat_stable = B_stable.dot(one), B_stable.T.dot(one)
            oper_num = oper_num + 2 * ( 2 * Size * Size - Size)
            
            Bs_stable = u_hat_stable.sum()
            oper_num = oper_num +  Size - 1
            
            phi_new = gamma*(-lamu_new[:Size].dot(conalpha) - lamu_new[Size:].dot(conbeta) + np.log(Bs_stable) + max_logB)
            oper_num = oper_num + 2*Size + 5
            
            grad_phi_new = gamma * np.concatenate((-conalpha + u_hat_stable/Bs_stable, -conbeta + v_hat_stable/Bs_stable),0)                       
            oper_num = oper_num + 6 * Size
                       
            z_new = z - alpha_new * grad_phi_new
            oper_num = oper_num + 4*Size     
            
            eta_new = tau * z_new + (1-tau) * eta
            oper_num = oper_num + 2*Size*3 + 1
            
            phi_eta = f(eta_new)
            oper_num = oper_num + 6*Size*Size+ 2*Size+4
            
            
            oper_num = oper_num + 3 + 10 *Size
            if phi_eta <= phi_new + grad_phi_new.dot(eta_new - lamu_new) + L * ((eta_new - lamu_new)**2).sum() / 2:
                
                betta = betta_new
                z = z_new.copy()
                eta = eta_new.copy()
                break    
            L = L * 2
            oper_num = oper_num + 1
          
        L= L/2
        oper_num = oper_num + 1
        
        primal_var = tau * B_stable/Bs_stable + (1 - tau) * primal_var
        oper_num= oper_num + 4*Size*Size + 1
        
        k=k+1
        


        valueot = np.dot(C.reshape(1,-1),primal_var.reshape(-1,1))[0][0]
        trueacc=abs(np.squeeze(primal_var.sum(axis=1).reshape(-1,1)) - conalpha).sum() + abs(np.squeeze(primal_var.sum(axis=0).reshape(-1, 1)) - conbeta).sum()
        Store_Acc.append(trueacc)
        Store_Oper.append(oper_num)
        Store_value.append(valueot)

          
        timecost = time.time() - time0
        
        Store_Time.append(timecost)
        if  trueacc <= acc:  
            return primal_var, oper_num, timecost

def APDAGD_w_rounding(epsilon, Size, C, alpha, beta):  
    penalty = epsilon / (4 * np.log(Size))
    epsilonprime = epsilon / 8 * abs(C).max()
    alphatilde = (1 - epsilonprime / 8) * alpha + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    betatilde = (1 - epsilonprime / 8) * beta + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    Xtilde, step, time= APDAGD(acc = epsilonprime / 2, conalpha = np.squeeze(alphatilde), conbeta = np.squeeze(betatilde), C = C, Size = Size, gamma= penalty)
    Xhat = roundto(Xtilde, alpha = alpha, beta = beta)
    return Xhat, step, time


#%%Sinkhorn
def sinkhorn_log(a, b, M, reg, acc):
    time0 = time.time()
    dim_a = len(a)
    dim_b = b.shape[0]
    Size = dim_a


    oper = 0

    Mr = - M / reg
    oper += Size

    u = np.zeros(dim_a)
    v = np.zeros(dim_b)

    def get_logT(u, v):

        return Mr + u[:, None] + v[None, :]

    loga = np.log(a)
    logb = np.log(b)

    error = 1
    while True:

        v = logb - scipy.special.logsumexp(Mr + u[:, None], 0)
        oper += Size + Size * Size*3

        u = loga - scipy.special.logsumexp(Mr + v[None, :], 1)
        oper += Size + Size * Size*3

        G = np.exp(get_logT(u, v))
        oper += 3*Size*Size




        error = abs(np.sum(G, axis=1) - a).sum() + abs(np.sum(G, axis=0) - b).sum()
        if error < acc:
            timecost = time.time() - time0
            return G, oper, timecost


def Sinkhorn_w_rounding(epsilon, Size, C, alpha, beta):
    penalty = epsilon / (4 * np.log(Size))
    epsilonprime = epsilon / 8 * abs(C).max()
    alphatilde = (1 - epsilonprime / 8) * alpha + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    betatilde = (1 - epsilonprime / 8) * beta + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    Xtilde, Iter, time = sinkhorn_log(a = np.squeeze(alphatilde), b = np.squeeze(betatilde), M = C, reg = penalty, acc = epsilonprime / 2)
    Xhat = roundto(Xtilde, alpha = alpha, beta = beta)   
    return Xhat, Iter, time

#%% AAM
def AAM(acc, Size, gamma, conalpha, conbeta, C):
    time0 = time.time()
    L=1
    step = 2
    one = np.ones(Size, np.float64)
    x0 = np.zeros(2*Size, np.float64)
    xi = np.zeros_like(x0)
    eta = xi.copy()
    zeta = xi.copy()
    eta_new = xi.copy()
    zeta_new = xi.copy()
    grad2 = alpha_new = alpha = 0
    ustep = np.zeros_like(x0[:Size])
    vstep = np.zeros_like(ustep)
    
    f = lambda x: phi_(x, gamma, C, Size, conalpha, conbeta)
    f_primal = lambda x: f_(gamma, x, C, Size)
    
    
    operation = 0 
    

    
    primal_var = np.zeros_like(C)
    
    
    
    Store_Acc=[]
    Store_Oper=[]
    Store_Time=[]
    Store_value = []
    valueotinitial = np.dot(C.reshape(1,-1),primal_var.reshape(-1,1))[0][0]
    Store_value.append(valueotinitial)
    acc_temp = abs(np.squeeze(primal_var.sum(axis=1).reshape(-1,1)) - conalpha).sum() + abs(np.squeeze(primal_var.sum(axis=0).reshape(-1, 1)) - conbeta).sum()
    Store_Acc.append(acc_temp)
    Store_Oper.append(0)   
    Store_Time.append(0)

    
    while True:
        
        
        L_new = L/step
        operation = operation + 1
        
        K=-C/gamma
        operation = operation +Size *Size
        while True:
            alpha_new = 1/2/L_new + np.sqrt(1/4/L_new/L_new + alpha*alpha*L/L_new)
            operation = operation + 11
            
            tau = 1/alpha_new/L_new
            operation = operation + 2
            
            xi = tau * zeta + (1 - tau) * eta
            operation = operation + 6 * Size + 1
            
            
            ##############
            logB = (K + np.outer(xi[:Size], one) + np.outer(one, xi[Size:]))
            operation = operation + 4 * Size * Size
            
            max_logB =logB.max()
            logB_stable = logB - max_logB
            operation = operation + Size*Size

            B_stable = np.exp(logB_stable)
            operation = operation + Size*Size
            
            u_hat_stable, v_hat_stable = B_stable.dot(one), B_stable.T.dot(one)
            operation = operation + 2 * ( 2 * Size * Size - Size)
            
            
            Bs_stable = u_hat_stable.sum()
            operation = operation + Size - 1
            

            f_xi = gamma*(-xi[:Size].dot(conalpha) - xi[Size:].dot(conbeta) + np.log(Bs_stable) + max_logB)
            operation = operation + 2*Size + 5
            
            grad_f_xi = gamma*np.concatenate((-conalpha + u_hat_stable/Bs_stable, -conbeta + v_hat_stable/Bs_stable),0)            
            operation = operation + 6 * Size
            
            
            gu, gv = (grad_f_xi[:Size]**2).sum(), (grad_f_xi[Size:]**2).sum()
            operation = operation + 4*Size -2
            
            norm2_grad_f_xi = (gu+gv)
            operation = operation + Size

            if gu > gv:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        ustep = conalpha/u_hat_stable
                        operation = operation + Size
                        
                    except Warning as e:
                        u_hat_stable/=u_hat_stable.max()
                        operation = operation + Size
                        
                        u_hat_stable[u_hat_stable<1e-150] = 1e-150
                        ustep = conalpha/u_hat_stable
                        operation = operation + Size
                        #print('catchu')
                    
                
                ustep/=ustep.max()
                operation = operation + Size
                
                xi[:Size]+=np.log(ustep)
                operation = operation + Size *2
                
                Z=ustep[:,None]*B_stable
                operation = operation + Size
                
                
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        vstep = conbeta/v_hat_stable
                        operation = operation + Size
                    except Warning as e:
                        v_hat_stable/=v_hat_stable.max()
                        operation = operation + Size
                        
                        v_hat_stable[v_hat_stable<1e-150] = 1e-150
                        vstep = conbeta/v_hat_stable
                        operation = operation + Size
                        

                vstep/=vstep.max()
                operation = operation +Size
                
                xi[Size:]+=np.log(vstep)
                operation = operation + Size*2
                
                Z=B_stable*vstep[None,:]
                operation = operation + 2*Size*Size -Size
                
            f_eta_new=gamma*(np.log(Z.sum())+max_logB-xi[:Size].dot(conalpha)-xi[Size:].dot(conbeta))
            operation = operation+ Size*Size +2*Size+4
            
            operation = operation + 3
            if f_eta_new <= f_xi - (norm2_grad_f_xi)/2/L_new: # can be optimized 2 itmes
                primal_var = (alpha_new * B_stable/Bs_stable + L * alpha**2 * primal_var) /(L_new*alpha_new**2)
                operation = operation + 4*Size*Size + 5
                
                zeta -= alpha_new * grad_f_xi
                operation = operation + 4*Size
                
                eta = xi.copy()
                alpha = alpha_new
                L = L_new
                
                break
            L_new*=step
            operation = operation + 1
            
            


        valueot = np.dot(C.reshape(1,-1),primal_var.reshape(-1,1))[0][0]
        
        trueacc=abs(np.squeeze(primal_var.sum(axis=1).reshape(-1,1)) - conalpha).sum() + abs(np.squeeze(primal_var.sum(axis=0).reshape(-1, 1)) - conbeta).sum()
        Store_Acc.append(trueacc)
        Store_Oper.append(operation)

        Store_value.append(valueot)
        
          
        timecost = time.time() - time0
        Store_Time.append(timecost)
        
        if  trueacc <= acc:        
            return primal_var, operation, timecost


def AAM_w_rounding(epsilon, Size, C, alpha, beta): 
    penalty = epsilon / (4 * np.log(Size))
    epsilonprime = epsilon / 8 * abs(C).max()
    alphatilde = (1 - epsilonprime / 8) * alpha + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    betatilde = (1 - epsilonprime / 8) * beta + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    Xtilde, oper, time = AAM(acc = epsilonprime / 2, conalpha = np.squeeze(alphatilde), conbeta = np.squeeze(betatilde), C = C, Size = Size, gamma= penalty)
    Xhat = roundto(Xtilde, alpha = alpha, beta = beta)
    return Xhat, oper, time


#%% APDRCD
def deltaphi(alpha, beta, C, Size, y, ik, penalty):
    if ik < Size:
       return np.exp( (-C[ik,:] + y[ik][0] + y[Size:].reshape(1,-1)) / penalty - 1 ).sum() - alpha[ik]
    
    if ik >= Size:
       return np.exp( (-C[:,(ik - Size)].reshape(-1,1) + y[ik][0] + y[:Size]) / penalty - 1 ).sum() - beta[ik - Size]


def primaldual(C, penalty, Size, y):
    return np.exp( (-C + y[:Size] + y[Size:].reshape(1,-1))/penalty - 1 )


def APDRCD(alpha, beta, C, penalty, Size, acc):
      time0 = time.time()
      seed = 123
      np.random.seed(seed) 
      y_store = np.zeros((2*Size, 1),np.float64)
      z_temp = np.zeros((2*Size, 1),np.float64)
      lambda_temp = np.zeros((2*Size, 1),np.float64) 
      x_s = np.zeros((Size, Size),np.float64)
      C_temp = 1
      D_temp = np.zeros((Size, Size),np.float64)
      #Lip = (1 / penalty)*max(alpha)*Size
      Lip = (4 / penalty)
      
      flag = 0 
      Store_Error = []
      Store_Flag = []
      Store_Time = []
      Store_value = []
      acc_temp = abs(x_s.sum(axis=1).reshape(-1,1) - alpha).sum() + abs(x_s.sum(axis=0).reshape(-1, 1) - beta).sum()
      valueotinitial = np.dot(C.reshape(1,-1),x_s.reshape(-1,1))[0][0]
      Store_Error.append(acc_temp)
      Store_Flag.append(0)
      Store_Time.append(0)
      Store_value.append(valueotinitial)
      timecost = 0
      error = 100
      theta = 1
      
      while error >= acc:
          y_store =  (1 - theta) * lambda_temp + theta * z_temp
          flag  = flag + 6 * Size + 1
         
            
         
          D_temp = D_temp + primaldual(C =C, penalty = penalty, Size = Size, y = y_store)/theta
          ## When transforming to the primal variable, we only need to update n elements per step, but for implementation convenience, we update n^2 elements per step.
          ## When computing the complexity, we still view the number of operations as n.
          ## Also notice that as we demenstrated in our paper, the addition of two matrices and divide a matrix by a constat will cost O(n^2)

          flag = flag + Size * Size + 5 * Size
          
          x_s = D_temp / C_temp
          flag = flag + Size * Size
         
          
          ik = np.random.choice(2 * Size)
          
          Deltaphi = deltaphi(alpha = alpha, beta = beta, C = C, Size = Size, y = y_store, ik = ik, penalty = penalty)
          flag  = flag + 6 * Size
          
          lambda_temp[ik] = y_store[ik] -  Deltaphi / Lip
          flag  = flag + 2
          
          
          z_temp[ik] = z_temp[ik] - Deltaphi / (2 * Size * Lip * theta)
          flag  = flag + 5
          
          
          C_temp = C_temp + 1/theta
          flag = flag + 2
          
          
         
          
          theta = (-theta**2 + np.sqrt( theta**4 + 4*theta**2 )   )/2
          flag = flag + 8
          
          
          error = abs(x_s.sum(axis=1).reshape(-1,1) - alpha).sum() + abs(x_s.sum(axis=0).reshape(-1, 1) - beta).sum()
          Store_Error.append(error)
          Store_Flag.append(flag)

          
          valueot = np.dot(C.reshape(1,-1),x_s.reshape(-1,1))[0][0]
          Store_value.append(valueot)
          
          
          timecost = time.time()-time0
          Store_Time.append(timecost)
          
          
      return x_s, flag, timecost


def APDRCD_w_rounding(epsilon, Size, C, alpha, beta):
    penalty = epsilon / (4 * np.log(Size))
    epsilonprime = epsilon / 8 * abs(C).max()
    alphatilde = (1 - epsilonprime / 8) * alpha + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    betatilde = (1 - epsilonprime / 8) * beta + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    Xtilde, Iter, time = APDRCD(alpha = alphatilde, beta = betatilde , Size = Size, 
           acc = epsilonprime / 2, 
           penalty = penalty, C = C)
    Xhat = roundto(Xtilde, alpha = alpha, beta = beta)   
    return Xhat, Iter, time

#%% Stochastic Sinkhorn
def stochastic_sinkhorn(a, b, M, reg, acc):
    time0 = time.time()
    dim_a = a.shape[0]
    dim_b = b.shape[0]
    Size = dim_a

    oper = 0

    Mr = -M / reg
    oper += Size*Size

    K = np.exp(Mr)
    oper += Size*Size

    u = np.full((dim_a,), 1. / dim_a)
    oper += 1

    v = np.full((dim_b,), 1. / dim_b)
    oper += 1

    logu = np.log(u)
    oper += Size

    logv = np.log(v)
    oper += Size

    G = u[:, None] * K * v[None, :]
    oper += Size*Size*2

    G_log = np.exp(G)
    oper += Size*Size

    viol = np.sum(G, axis=1) - a
    oper += + Size*Size

    viol_2 = np.sum(G, axis=0) - b
    oper += + Size*Size



    while True:

        v = np.concatenate((viol,viol_2))
        p = np.abs(v)/np.abs(v).sum()
        j = np.random.multinomial(1, p.reshape(-1)).argmax()

        if j < Size:
            i_1 = j
            old_u_log = logu[i_1]

            new_u_log = np.log(a[i_1]) - scipy.special.logsumexp(Mr[i_1, :]+logv)
            oper += 1+1+Size+Size+Size


            G_log[i_1,:] = np.full(dim_a,new_u_log) +  Mr[i_1,:] + logv
            oper += Size*2


            viol[i_1]= np.exp(new_u_log + scipy.special.logsumexp(Mr[i_1, :]+logv))- a[i_1]
            oper += 1+1+1+Size+Size+Size

            viol_2 += np.exp(Mr[i_1, :].T + new_u_log + logv) - np.exp(Mr[i_1, :].T + old_u_log + logv)
            oper += Size*3 + Size + Size*3 + Size

            logu[i_1] = new_u_log
        else:
            i_2 = j -Size
            old_v_log = logv[i_2]

            new_v_log = np.log(b[i_2]) - scipy.special.logsumexp(Mr[:, i_2]+ logu)
            oper += 1+1+Size+Size+Size


            G_log[:,i_2] =   Mr[:,i_2] + logu + np.full(dim_a,new_v_log)
            oper += Size*2

            viol += np.exp(new_v_log + Mr[:, i_2] +logu) - np.exp(old_v_log + Mr[:, i_2] +logu)
            oper += 1+1+1+Size+Size+Size

            viol_2[i_2] = np.exp(new_v_log + scipy.special.logsumexp(Mr[:, i_2]+logu))- b[i_2]
            oper += Size*3 + Size + Size*3 + Size

            logv[i_2] = new_v_log

        G = np.exp(G_log) 
        ##We only need to update one column/one row here. But for convience, we implement to upadate all the elements. 
        ##When we calculate the number of operations, we only add n here.
        oper += Size
        error = abs(np.sum(G, axis=1) - a).sum() + abs(np.sum(G, axis=0) - b).sum()

        if error<=acc:
            return G,oper, time.time() - time0

 

def Stochastic_sinkhorn_w_rounding(epsilon, Size, C, alpha, beta):  
    penalty = epsilon / (4 * np.log(Size))
    epsilonprime = epsilon / 8 * abs(C).max()
    alphatilde = (1 - epsilonprime / 8) * alpha + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    betatilde = (1 - epsilonprime / 8) * beta + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    Xtilde, Iter, time = stochastic_sinkhorn(a = np.squeeze(alphatilde), b = np.squeeze(betatilde), M = C, reg = penalty, acc = epsilonprime / 2)
    Xhat = roundto(Xtilde, alpha = alpha, beta = beta)
    return Xhat, Iter, time








