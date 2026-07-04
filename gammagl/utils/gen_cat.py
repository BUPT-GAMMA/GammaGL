# This code is from: [GenCAT](https://github.com/seijimaekawa/GenCAT)
import numpy as np
from numpy import linalg as la
from scipy import sparse
import random
import copy
import sys
import powerlaw

import warnings
warnings.simplefilter('ignore')


def config_diagonal(M, D, x=1):
    import copy
    k = M.shape[0]  # number of classes
    M_ = copy.deepcopy(M)
    D_ = copy.deepcopy(D)
    if x != 0:
        for i in range(k):  # for each diagonal element
            # for i in range(int(k/2)+1,k):
            for j in range(k):
                if i == j:
                    M_[i][j] -= 0.1 * x
                else:
                    M_[i][j] += (0.1 * x) / (k - 1)
            M_[M_ < 0] = 0
        for i in range(k):
            # for i in range(int(k/2)+1,k):
            for j in range(k):
                if i == j:
                    D_[i][j] = D_[i][j] * (M_[i][j] / (M_[i][j] + 0.1 * x))
                else:
                    D_[i][j] = D_[i][j] * (M_[i][j] / (M_[i][j] - (0.1) / (k - 1)))
        for i in range(k):
            M_[i] = M_[i] / sum(M_[i])
        D_[D_ <= 0] = 0
    return M_, D_


def feature_extraction(S, X, Label):
    k = max(Label) + 1
    M, D = calc_class_features(S, k, Label)
    H = calc_attr_cor(X, Label)

    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(Label)):
        partition[Label[i]].append(i)

    class_size = []
    for i in partition:
        class_size.append(len(i))
    class_size = np.array(class_size) / sum(class_size)

    # node degree
    theta = np.zeros(len(Label))
    nnz = S.nonzero()
    for i in range(len(nnz[0])):
        if nnz[0][i] < nnz[1][i]:
            theta[nnz[0][i]] += 1
            theta[nnz[1][i]] += 1

    return M, D, list(class_size), H, sorted(theta, reverse=True)


def calc_class_features(S, k, Label):
    pref = np.zeros((len(Label), k))
    nnz = S.nonzero()
    for i in range(len(nnz[0])):
        if nnz[0][i] < nnz[1][i]:
            pref[nnz[0][i]][Label[nnz[1][i]]] += 1
            pref[nnz[1][i]][Label[nnz[0][i]]] += 1
    for i in range(len(Label)):
        pref[i] /= sum(pref[i])
    pref = np.nan_to_num(pref)

    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(Label)):
        partition[Label[i]].append(i)

    # caluculate average and deviation of class preference
    from statistics import mean, median, variance, stdev
    class_pref_mean = np.zeros((k, k))
    class_pref_dev = np.zeros((k, k))
    for i in range(k):
        pref_tmp = []
        for j in partition[i]:
            pref_tmp.append(pref[j])
        pref_tmp = np.array(pref_tmp).transpose()
        for h in range(k):
            class_pref_mean[i, h] = mean(pref_tmp[h])
            if len(pref_tmp[h]) > 1:
                class_pref_dev[i, h] = stdev(pref_tmp[h])
            else:
                class_pref_dev[i, h] = 0
    return class_pref_mean, class_pref_dev


def calc_attr_cor(X, Label):
    k = max(Label) + 1
    n = X.shape[0]
    d = X.shape[1]

    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(Label)):
        partition[Label[i]].append(i)

    from statistics import mean
    attr_cor = np.zeros((d, k))
    for i in range(k):
        tmp = np.zeros(d)
        for j in partition[i]:
            tmp += X[j]
        attr_cor[:, i] = tmp / len(partition[i])
    return attr_cor


def node_deg(n,m,max_deg,p=3.):
    simulated_data = [0]
    while sum(simulated_data)/2 < m:
        theoretical_distribution = powerlaw.Power_Law(xmin = 1., parameters = [p])
        simulated_data=theoretical_distribution.generate_random(n)
        over_list = np.where(simulated_data>max_deg)[0]
        while len(over_list) != 0:
            add_deg = theoretical_distribution.generate_random(len(over_list))
            for i,node_id in enumerate(over_list):
                simulated_data[node_id] = add_deg[i]
            over_list = np.where(simulated_data>max_deg)[0]
        simulated_data = np.round(simulated_data)
        if (m - sum(simulated_data)/2) < m/5:
            p -= 0.01
        else:
            p -= 0.1
        if p<1.01:
            print("break")
            break
    # print("expected number of edges : ",int(sum(simulated_data)/2))
    return sorted(simulated_data,reverse=True)


def count_node_degree(S):
    n = S.shape[0]
    node_degree = np.zeros(n)
    nnz = S.nonzero()
    for i in range(len(nnz[0])):
        if nnz[0][i] < nnz[1][i]:
            node_degree[nnz[0][i]] += 1
            node_degree[nnz[1][i]] += 1
    return int(sum(node_degree)/2)

def distribution_generator(flag, para_pow, para_normal, para_zip, t):
    if flag == "power_law":
        dist = 1 - np.random.power(para_pow, t) # R^{k}
    elif flag == "uniform":
        dist = np.random.uniform(0,1,t)
    elif flag == "normal":
        dist = np.random.normal(0.5,para_normal,t)
    elif flag == "zipfian":
        dist = np.random.zipf(para_zip,t)
    return dist

def class_size_gen(k,phi_c):
    # chi = distribution_generator("power_law",phi_c,0,0, k)
    chi = distribution_generator("normal",phi_c,0,0, k)
    return np.array(chi) / sum(chi)


def latent_factor_gen(n,k,M,D,class_size):
    import sys
    density = np.zeros(k)
    for l in range(k):
        density[l] = M[l,l]

    # generate U from class preference matrix
    def reverse(U_tmp,l):
        U_ = 1 - U_tmp
        sum_U_ = sum(U_) - U_tmp[l]
        for i in range(k):
            if i != l:
                U_[i] = U_[i] * U_tmp[l] / sum_U_
        return U_

    U = np.zeros((n,k))
    C=[]
    for i in range(n):
        C_tmp = random.choices(list(range(0,k)),k=1,weights=class_size)[0]
        C.append(C_tmp)
        for h in range(k):
            U[i,h] = np.random.normal(loc=M[C_tmp][h],scale=D[C_tmp][h],size=1)[0]
        if M[C_tmp,C_tmp] < 1/k: # for heterophily
            U[i] = reverse(U[i],C_tmp)

    # eliminate U<0 and U>1 (keep 0<=U<=1)
    minus_list = np.where(U <= 0)
    for i in range(len(minus_list[0])):
        U[minus_list[0][i],minus_list[1][i]] = sys.float_info.epsilon
    one_list = np.where(U > 1)
    for i in range(len(one_list[0])):
        U[one_list[0][i],one_list[1][i]] = 1
    # normalize
    for i in range(n):
        U[i] /= sum(U[i])

    return U,C,density

def adjust(n,k,U,C,M): # fitting latent factors to given statistics by minimizing loss
    U_prime = copy.deepcopy(U)
    partition = []
    for l in range(k):
        partition.append([])
    for i in range(len(C)):
        partition[C[i]].append(i)

    # Freezing function
    def freez_func(q,Th):
        return q**(1/Th) / np.sum(q**(1/Th))

    # Reverse function
    def reverse(U_tmp,l):
        U_ = 1 - U_tmp
        sum_U_ = sum(U_) - U_tmp[l]
        for i in range(k):
            if i != l:
                U_[i] = U_[i] * U_tmp[l] / sum_U_
        return U_
    flag=0
    for l in range(k):
        loss_min = float('inf')
        if  M[l][l] >= 1/k:
            for Th in np.arange(0.01,1,0.01):
                sum_estimated = np.zeros(k)
                for i in partition[l]:
                    sum_estimated += freez_func(U[i],Th) * freez_func(U[i],Th)
                loss_tmp = la.norm(M[l]-sum_estimated/len(partition[l]))
                if loss_tmp < loss_min:
                    loss_min = loss_tmp
                    Th_min = Th
            for i in partition[l]:
                U[i] = freez_func(U[i],Th_min)
                U_prime[i] = U[i]
        else:
            for Th in np.arange(0.01,1,0.01):
                sum_estimated = np.zeros(k)
                for i in partition[l]:
                    sum_estimated += freez_func(U[i],Th) * reverse(freez_func(U[i],Th),l)
                loss_tmp = la.norm(M[l]-sum_estimated/len(partition[l]))
                if loss_tmp < loss_min:
                    loss_min = loss_tmp
                    Th_min = Th
            for i in partition[l]:
                U[i] = freez_func(U[i],Th_min)
                U_prime[i] = reverse(U[i],l)
#         print(Th_min)
    return U, U_prime

def edge_construction(n, U, k, U_prime, step, theta, r):
    U_ = copy.deepcopy(U)

    S = sparse.dok_matrix((n,n))
    degree_list = np.zeros(n)
    count_list = []

    print_count = 1
    for i in range(n): # for each node
#         if i/n * 10 > print_count:
#             print("finished " +str(print_count)+"0%")
#             print_count += 1
        count = 0
        ng_list = set([i]) # store how many loops edge generation needs. This is used to monitoring the behavior of GenCAT, not an algorithmic part.
        while count < r and degree_list[i] < theta[i]:
            to_classes = random.choices(list(range(0,k)), k=int(theta[i]-degree_list[i]), weights=U_[i]) # choose classes based on U[i]
            for to_class in to_classes:
                for loop in range(50):
                    j = U_prime[to_class][int(random.random()/step)] # choose nodes from to_class based on the probability of U'[to_class]
                    if j not in ng_list: # if node i and node j do not have an edge
                        ng_list.add(j)
                        break
                if degree_list[j] < theta[j] and i!=j:
                    S[i,j] = 1;S[j,i] = 1 # undirected edge
                    degree_list[i]+=1;degree_list[j]+=1
            count += 1
        count_list.append(count) # store how many loops edge generation needs. This is used to monitoring the behavior of GenCAT, not an algorithmic part.
    return S, count_list

def ITS_U_prime(n,k,U_prime,step): # inverse transformation sampling for efficient edge generation
    class_list = []
    UT = U_prime.transpose()
    for i in range(k): # create a cumulative distribution for each class
        UT_tmp = UT[i]/ sum(UT[i])
        for j in range(n-1):
            UT_tmp[j+1] += UT_tmp[j]

        class_tmp = []
        node_counter = 0
        for l in np.arange(0,1,step):
            if node_counter >= n-1:
                class_tmp.append(n-1)
            elif UT_tmp[node_counter] > l:
                class_tmp.append(node_counter)
            else:
                node_counter += 1
                class_tmp.append(node_counter)
        class_list.append(class_tmp)
    return class_list


def adjust_att(n,k,d,U,C,H):
    V = copy.deepcopy(H)
    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(C)):
        partition[C[i]].append(i)

    # Freezing function
    def freez_func(q,Th):
        return q**(1/Th) / np.sum(q**(1/Th))

    P = np.zeros((k,k))
    for l in range(k):
        for j in partition[l]:
            P[l] += U[j]
        P[l] = P[l]/len(partition[l])

    for delta in range(d):
        loss = []
        for Th in np.arange(0.1, 1.1, 0.05):
            loss.append(np.linalg.norm(H[delta] - P @ freez_func(V[delta],Th).T))
        V[delta] = freez_func(V[delta],0.1*(np.argmin(loss)+1))
    return V

def attribute_generation(n,d,k,U,V,C,omega,att_type,att_scale):
    X = U@V.T

    if att_type == "normal":
        for i in range(d): # each attribute demension
            clus_dev = np.random.uniform(omega*0.99,omega*1.01,k) # variation for each class
            for p in range(n): # each node
                X[p,i] += np.random.normal(0.0,clus_dev[C[p]],1)
        # normalization [0,1]
        for i in range(d):
            X[:,i] -= min(X[:,i])
            X[:,i] /= max(X[:,i])
    elif att_type == "Bernoulli": # Bernoulli distribution
        for i in range(d):
            X[:,i] = X[:,i] * att_scale[i]
        X = X - np.random.rand(n,d) # efficient approach to compute probabilities
        X[X>=0] = 1
        X[X<0] = 0
    else:
        raise NotImplemented
    return np.nan_to_num(X)


#####################################################################################
#####################################################################################
############################ modules for ablation study ###################################
#####################################################################################
#####################################################################################
# These modules are not used for the functionality of FULL GenCAT.

def adjust_woAP(n,k,U,C,density): # skip adjusting phases
    U_prime = copy.deepcopy(U)
    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(C)):
        partition[C[i]].append(i)

    def reverse(U_tmp,l,k):
        U_ = 1 - U_tmp
        sum_U_ = sum(U_) - U_tmp[l]
        for i in range(k):
            if i != l:
                U_[i] = U_[i] * U_tmp[l] / sum_U_
        return U_
    for l in range(k):
        if  density[l] < 1/k: # for heterphilic classes
            for j in partition[l]:
                U_prime[i] = reverse(U[j],l,k)
    return U_prime

def edge_construction_wo_ITS(n, U, k, U_primeT, theta, r): # not using inverse transform sampling for U'
    S = sparse.dok_matrix((n,n))
    degree_list = np.zeros(n)
    count_list = []

    print_count = 1
    reconst = U @ U_primeT # simply construct probability matrix
    for i in range(n):
        count = 0
        ng_list = set([i]) # A temporal set storing node IDs that have already connected with node i. This set improves the efficiency.

        while count < r and degree_list[i] < theta[i]:
            to_nodes = random.choices(list(range(0,n)), k=int(theta[i]-degree_list[i]), weights=reconst[i])
            for j in to_nodes:
                if degree_list[j] < theta[j] and i!=j:
                    S[i,j] = 1;S[j,i] = 1
                    degree_list[i]+=1;degree_list[j]+=1
            count += 1
        count_list.append(count) # store how many loops edge generation needs. This is used to monitoring the behavior of GenCAT, not an algorithmic part.
    return S, count_list


##########################################################################################
##########################################################################################
################################# main function ##########################################
##########################################################################################
##########################################################################################


def gencat(M, # class preference mean : R^{kxk}
            D, # class preference deviation : R^{kxk}
            H, # attribute-class correlation : R^{dxk}
            class_size=None, # class size distribution : R^k
            n=3000, # the number of nodes : integer
            m=5000, # expected number of edges : integer
            p=3., # parameter of power-law distribution : float
            max_deg=None, # upper bound of node degree : integer
            theta=None, # node degree distribution : R^n. Users need to set either "n and m" or "theta".
            phi_c=1, # parameter of power-low distribution for class size : float. Users need to set either "class_size" or "phi_c".
            omega=0.2, # deviation of normal distribution for attribute generation : float
            r=20, # the number of iterations for generating edges : integer
            step=50, # the number of steps for inverse transform sampling : integer
            att_type="normal", # attribute type : [Bernoulli, normal]
            woAP=False,woITS=False):
    k=M.shape[0]
    d=H.shape[0]

    if theta == None:
      # node degree generation
      if max_deg == None:
        max_deg = int(n/10)
      theta = node_deg(n,m,max_deg,p)
    else:
      n = len(theta)
      m = int(sum(theta)/2)
#     line_warn(sum(theta)/2)

    # class generation
    if class_size == None:
        class_size = class_size_gen(k,phi_c)
    U,C,density = latent_factor_gen(n,k,M,D,class_size)

    # adjusting phase
    if not woAP:
        U,U_prime = adjust(n,k,U,C,M)
    else:
        print("woAP")
        U_prime = adjust_woAP(n,k,U,C,M)

    # Inverse Transform Sampling
    if not woITS:
        step = 1/(n*step)
        U_prime_CDF = ITS_U_prime(n,k,U_prime,step)

        # Edge generation
        S_gen, count_list = edge_construction(n, U, k, U_prime_CDF, step, theta, r)
    else:
        print("woITS")
        S_gen, count_list = edge_construction_wo_ITS(n, U, k, U_prime.T, theta, r)

    # print("number of generated edges : " + str(count_node_degree(S_gen)))

    ### Attribute
    att_scale = np.sum(H, axis=1)
    H_ = copy.deepcopy(H)
    for i in range(d):
        H_[i] = H_[i] / sum(H_[i])

    # adjust attribute-class proportion
    V= adjust_att(n,k,d,U,C,H_)

    # Attribute generation
    X = attribute_generation(n,d,k,U,V,C,omega,att_type,att_scale)

    return S_gen,X,C

##########################################################################################
##########################################################################################
############################## for simple input ##########################################
##########################################################################################
##########################################################################################

def gencat_simple(n,m,density,H,class_size=None,max_deg=None,p=3.,theta=None,phi_c=1,omega=0.2,r=50,step=100,att_type="normal"):
    d=H.shape[0]
    k=len(density)
    # node degree generation
    if theta == None:
      # node degree generation
      if max_deg == None:
        max_deg = int(n/10)
      theta = node_deg(n,m,max_deg,p)
    else:
      n = len(theta)
      m = int(sum(theta)/2)

    # generate class preference mean from given diagonal elements
    M = np.zeros((k,k))
    for l1 in range(k):
        for l2 in range(k):
            if l1==l2:
                M[l1][l2] = density[l1]
            else:
                M[l1][l2] = (1-density[l1]) / (k-1)


    # class generation
    if class_size==None:
        U,C = class_generation(n,k,phi_c)

    # adjusting phase
    U,U_prime = adjust(n,k,U,C,M)

    # Inverse Transform Sampling
    step = 1/(n*step)
    U_prime_CDF = ITS_U_prime(n,k,U_prime,step)

    # Edge generation
    S_gen, count_list = edge_construction(n, U, k, U_prime_CDF, step, theta, r)

    # Attribute
    att_scale = np.sum(H, axis=1)
    H_ = copy.deepcopy(H)
    for i in range(d):
        H_[i] = H_[i] / sum(H_[i])
    V = adjust_att(n,k,d,U,C,H_)

    # Attribute generation
    X = attribute_generation(n,d,k,U,V,C,omega,att_type,att_scale)

    return S_gen,X,C

def class_generation(n, k, phi_c):
    class_size = class_size_gen(k,phi_c)

    U = np.random.dirichlet(class_size, n)
    C = [] # class assignment list (finally, R^{n})
    for i in range(n):
        C.append(np.argmax(U[i]))

    counter=[];x=[]
    for i in range(k):
        x.append(i)
        counter.append(C.count(i))
    print("class size disribution : ",end="")
    print(counter)
    if 0 in counter:
        print('Error! There is a class which has no member.')
        sys.exit(1)

    return U,C

##########################################################################################
##########################################################################################
############################## for reproduction ##########################################
##########################################################################################
##########################################################################################

def class_reproduction(k,S,Label):
    # extract class preference matrix from given graph

    M, D = calc_class_features(S,k,Label)

    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(Label)):
        partition[Label[i]].append(i)

    class_size = []
    for i in partition:
        class_size.append(len(i))
    class_size = np.array(class_size) / sum(class_size)
    return M,D,class_size

def gencat_reproduction(S,Label,X=None,H=None,n=0,m=0,max_deg=0,omega=0.2,r=50,step=100,att_type="Bernoulli"):
    if X == None and H == None:
        d=0
    elif X != None:
        from func import calc_attr_cor
        H = calc_attr_cor(X,Label)
        d = H.shape[0]
    else:
        d = H.shape[0]

    # node degree generation
    if n == 0:
        theta = np.zeros(len(Label))
        nnz = S.nonzero()
        for i in range(len(nnz[0])):
            if nnz[0][i] < nnz[1][i]:
                theta[nnz[0][i]] += 1
                theta[nnz[1][i]] += 1
    else:
        theta = node_deg(n,m,max_deg)
    n = len(theta)
    m = count_node_degree(S)
    k = len(set(Label))
    step = 1/(n*step)

    # class feature extraction
    M,D,class_size = class_reproduction(k,S,Label)

    # latent factor generation
    U,C,density = latent_factor_gen(n,k,M,D,class_size)

    # adjusting phase
    U,U_prime = adjust(n,k,U,C,M)

    # Inverse Transform Sampling
    U_prime_CDF = ITS_U_prime(n,k,U_prime,step)

    # Edge generation
    S_gen, count_list = edge_construction(n, U, k, U_prime_CDF, step, theta, r)
    print("number of generated edges : " + str(count_node_degree(S_gen)))

    ### Attribute
    if d != 0:
        att_scale = np.sum(H, axis=1)
        H_ = copy.deepcopy(H)
        for i in range(d):
            H_[i] = H_[i] / sum(H_[i])

        V = adjust_att(n,k,d,U,C,H_)

        # Attribute generation
        X = attribute_generation(n,d,k,U,V,C,omega,att_type,att_scale)
    else:
        X=[]

    return S_gen,X,C


##########################################################################################
##########################################################################################
############################## only attribute ############################################
##########################################################################################
##########################################################################################


def gencat_only_att(n,M,D,H,phi_c=1,omega=0.2,r=50,step=100,att_type="normal",woAP=False,woITS=False):
    k=M.shape[0]
    d=H.shape[0]
    # class generation
    class_size = class_size_gen(k,phi_c)
    U,C,density = latent_factor_gen(n,k,M,D,class_size)

    # adjusting phase
    if not woAP:
        U,U_prime = adjust(n,k,U,C,M)
    else:
        print("woAP")
        U_prime = adjust_woAP(n,k,U,C,M)

    S_gen = []

    ### Attribute
    att_scale = np.sum(H, axis=1)
    H_ = copy.deepcopy(H)
    for i in range(d):
        H_[i] = H_[i] / sum(H_[i])
    V = adjust_att(n,k,d,U,C,H_)

    # Attribute generation
    X = attribute_generation(n,d,k,U,V,C,omega,att_type,att_scale)

    # not applying user-specified distribution
    X_not = U@V.T
    for i in range(d):
        X_not[:,i] -= min(X_not[:,i])
        X_not[:,i] /= max(X_not[:,i])

    return S_gen,X,X_not,C
