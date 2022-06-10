modes = ["gen", "dis"]

# training settings
batch_size_gen = 64  # batch size for the generator
batch_size_dis = 64  # batch size for the discriminator
lambda_gen = 1e-5  # l2 loss regulation weight for the generator
lambda_dis = 1e-5  # l2 loss regulation weight for the discriminator
n_sample_gen = 20  # number of samples for the generator
lr_gen = 1e-3  # learning rate for the generator
lr_dis = 1e-3  # learning rate for the discriminator
n_epochs = 20  # number of outer loops
n_epochs_gen = 30  # number of inner loops for the generator
n_epochs_dis = 30  # number of inner loops for the discriminator
gen_interval = n_epochs_gen  # sample new nodes for the generator for every gen_interval iterations
dis_interval = n_epochs_dis  # sample new nodes for the discriminator for every dis_interval iterations
update_ratio = 1    # updating ratio when choose the trees

# model saving
load_model = False  # whether loading existing model for initialization
save_steps = 10

# other hyper-parameters
n_emb = 50
multi_processing = True  # whether using multi-processing to construct BFS-trees
window_size = 2

# application and dataset settings
app = "link_prediction"
dataset = "CA-GrQc"

# path settings
train_filename = "../../gan_datasets/" + app + "/" + dataset + "_train.txt"
test_filename = "../../gan_datasets/" + app + "/" + dataset + "_test.txt"
test_neg_filename = "../../gan_datasets/" + app + "/" + dataset + "_test_neg.txt"

pretrain_emb_filename_d = "../../gan_datasets/" + app + "/" + dataset + "_pre_train.emb"
pretrain_emb_filename_g = "../../gan_datasets/" + app + "/" + dataset + "_pre_train.emb"

emb_filenames = ["../../gan_results/" + app + "/" + dataset + "_gen_.emb",
                 "../../gan_results/" + app + "/" + dataset + "_dis_.emb"]
result_filename = "../../gan_results/" + app + "/" + dataset + ".txt"
cache_filename = "../../gan_cache/" + dataset + ".pkl"
model_log = "checkpoint/"
