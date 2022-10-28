import tensorlayerx as tlx
import numpy as np
import os


def write_embeddings_to_file(GANModel, args, choice):
    """write embeddings of the G and the D to files

    Args:
        GANModel:  GANModel model
        args: parameters setting
        choice:
            choice==1: write embeddings of G and D in files
            choice==2: write embeddings of G with the best accuracy in CA-GrQc_best_acc_gen_.emb
            choice==3: write embeddings of D with the best accuracy in CA-GrQc_best_acc_dis_.emb

    """
    modes = [GANModel.generator, GANModel.discriminator]
    emb_filenames = [f'{args.emb_folder}/CA-GrQc_gen_.emb',
                     f'{args.emb_folder}/CA-GrQc_dis_.emb']
    best_acc_emb_filenames = [f'{args.best_acc_emb_folder}/CA-GrQc_best_acc_gen_.emb',
                              f'{args.best_acc_emb_folder}/CA-GrQc_best_acc_dis_.emb']
    if choice == 1:
        for i in range(2):
            if tlx.BACKEND == 'torch':
                embedding_matrix = modes[i].embedding_matrix.detach()
            else:
                embedding_matrix = modes[i].embedding_matrix
            index = np.array(range(GANModel.n_node)).reshape(-1, 1)
            embedding_matrix = np.hstack([index, embedding_matrix])
            embedding_list = embedding_matrix.tolist()
            embedding_str = [str(int(emb[0])) + "\t" + "\t".join([str(x) for x in emb[1:]]) + "\n"
                             for emb in embedding_list]

            if not os.path.isdir(args.emb_folder):
                os.makedirs(args.emb_folder)

            with open(emb_filenames[i], "w+") as f:
                lines = [str(GANModel.n_node) + "\t" +
                         str(args.n_emb) + "\n"] + embedding_str
                f.writelines(lines)
    else:
        for i in range(2):
            if (choice == 2 and i == 0) or (choice == 3 and i == 1):
                if tlx.BACKEND == 'torch':
                    embedding_matrix = modes[i].embedding_matrix.detach()
                else:
                    embedding_matrix = modes[i].embedding_matrix
                index = np.array(range(GANModel.n_node)).reshape(-1, 1)
                embedding_matrix = np.hstack([index, embedding_matrix])
                embedding_list = embedding_matrix.tolist()
                embedding_str = [str(int(emb[0])) + "\t" + "\t".join([str(x) for x in emb[1:]]) + "\n"
                                 for emb in embedding_list]

                if not os.path.isdir(args.best_acc_emb_folder):
                    os.makedirs(args.best_acc_emb_folder)

                with open(best_acc_emb_filenames[i], "w+") as f:
                    lines = [str(GANModel.n_node) + "\t" +
                             str(args.n_emb) + "\n"] + embedding_str
                    f.writelines(lines)
