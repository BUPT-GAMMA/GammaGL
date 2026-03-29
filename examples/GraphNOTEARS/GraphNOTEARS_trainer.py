from sklearn.metrics import f1_score
from gammagl.models import GraphNOTEARS

def main():
    # w_graph_types = ['ER', 'BA']
    # p_graph_types = ['ER', 'SBM']
    # sem_types = ['gauss', 'exp', 'gumbel', 'uniform', 'logistic']

    # n=100
    # d=5
    # s0=1*d
    # w_graph_types='ER'
    # p_graph_types='ER'
    # sem_types='exp'
    # X,adj1,w_true, w_mat, p1_true, p1_mat=GraphNOTEARS.data_pre_p1(n,d,s0,w_graph_types,p_graph_types,sem_types)
    #
    # model1=GraphNOTEARS.model_p1_MLP(dims=[d,n,1],bias=True)
    # adj1_torch,X_torch=GraphNOTEARS.convert_Tensor_p1(adj1,X)
    # w_est,p1_est=GraphNOTEARS.p1_linear_model(model1,X_torch,adj1_torch,lambda1=0.01,lambda2=0.01,lambda3=0.01)
    #
    # thre=0.3
    # w_est=GraphNOTEARS.convert_by_threshold(w_est,thre)
    #
    # fdr, tpr, fpr, shd, pred_size = GraphNOTEARS.count_accuracy(w_true, w_est!= 0)
    # w_f1_ = f1_score(w_true, w_est != 0, average="micro")
    # acc = ' fdr = ' + str(fdr) + ' tpr = ' + str(tpr) + ' fpr = ' + str(
    #     fpr) + ' shd = ' + str(shd) + ' nnz = ' + str(pred_size)
    # string = "W_est: threshold = " + str(thre) + " acc : " + str(acc) + "  f1 = " + str(
    #     w_f1_) + "\n"
    #
    # print(string)

    n = 100
    d = 5
    s0 = 1 * d
    w_graph_types = 'ER'
    p_graph_types = 'SBM'
    sem_types = 'gauss'
    X, adj1, adj2, w_true, w_mat, p1_true, p1_mat, p2_true, p2_mat = GraphNOTEARS.data_pre_p2(n, d, s0, w_graph_types,
                                                                                              p_graph_types,
                                                                                              sem_types)
    model2 = GraphNOTEARS.model_p2_MLP(dims=[d, n, 1], bias=True)
    adj1_torch, adj2_torch, X_torch = GraphNOTEARS.convert_Tensor_p2(adj1, adj2, X)
    w_est, p1_est, p2_est = GraphNOTEARS.p2_linear_model(model2, X_torch, adj1_torch, adj2_torch, lambda1=0.01,
                                                         lambda2=0.01, lambda3=0.01)
    thre = 0.3
    w_est = GraphNOTEARS.convert_by_threshold(w_est, thre)

    fdr, tpr, fpr, shd, pred_size = GraphNOTEARS.count_accuracy(w_true, w_est != 0)
    w_f1_ = f1_score(w_true, w_est != 0, average="micro")
    acc = ' fdr = ' + str(fdr) + ' tpr = ' + str(tpr) + ' fpr = ' + str(
        fpr) + ' shd = ' + str(shd) + ' nnz = ' + str(pred_size)
    string = "W_est: threshold = " + str(thre) + " acc : " + str(acc) + "  f1 = " + str(
        w_f1_) + "\n"

    print(string)

if __name__=="__main__":
    main()