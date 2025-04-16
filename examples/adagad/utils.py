import yaml
import argparse
from gammagl.datasets import Planetoid
from gammagl.models import GCNModel
from gammagl.utils import add_self_loops, mask_to_index
from tensorlayerx.model import TrainOneStep, WithLoss

# from adagad import PreModel, ReModel
from gammagl.models import PreModel, ReModel

# build model
def build_pre_model(args):
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    num_hidden = args.num_hidden

    node_encoder_num_layers=args.node_encoder_num_layers
    edge_encoder_num_layers=args.edge_encoder_num_layers
    subgraph_encoder_num_layers=args.subgraph_encoder_num_layers

    attr_decoder_num_layers=args.attr_decoder_num_layers
    struct_decoder_num_layers=args.struct_decoder_num_layers

    residual = args.residual
    attn_drop = args.attn_drop
    in_drop = args.in_drop
    norm = args.norm
    negative_slope = args.negative_slope
    
    attr_encoder_type = args.attr_encoder
    struct_encoder_type = args.struct_encoder
    topology_encoder_type=args.topology_encoder

    attr_decoder_type = args.attr_decoder
    struct_decoder_type=args.struct_decoder

    replace_rate = args.replace_rate


    activation = args.activation
    loss_fn = args.loss_fn
    alpha_l = args.alpha_l
    concat_hidden = args.concat_hidden
    num_features = args.num_features

    mask_rate1 = args.mask_rate1
    drop_edge_rate1=args.drop_edge_rate1
    drop_path_rate1=args.drop_path_rate1
    predict_all_node1=args.predict_all_node1
    predict_all_edge1=args.predict_all_edge1
    drop_path_length1=args.drop_path_length1
    walks_per_node1=args.walks_per_node1

    mask_rate2 = args.mask_rate2
    drop_edge_rate2=args.drop_edge_rate2
    drop_path_rate2=args.drop_path_rate2
    predict_all_node2=args.predict_all_node2
    predict_all_edge2=args.predict_all_edge2
    drop_path_length2=args.drop_path_length2
    walks_per_node2=args.walks_per_node2

    mask_rate3 = args.mask_rate3
    drop_edge_rate3=args.drop_edge_rate3
    drop_path_rate3=args.drop_path_rate3
    predict_all_node3=args.predict_all_node3
    predict_all_edge3=args.predict_all_edge3
    drop_path_length3=args.drop_path_length3
    walks_per_node3=args.walks_per_node3

    select_gano_num=args.select_gano_num
    sano_weight=args.sano_weight

    attr_model = PreModel(
        in_dim=int(num_features),
        num_hidden=int(num_hidden),
        encoder_num_layers=node_encoder_num_layers,
        attr_decoder_num_layers=attr_decoder_num_layers,
        struct_decoder_num_layers=struct_decoder_num_layers,
        feat_drop=in_drop,
        mask_rate=mask_rate1,
        replace_rate=replace_rate,
        drop_edge_rate=drop_edge_rate1,
        drop_path_rate=drop_path_rate1,
        predict_all_edge=predict_all_edge1,
        drop_path_length=drop_path_length1,
        walks_per_node=walks_per_node1,
        select_gano_num=select_gano_num
    )

    struct_model = PreModel(
        in_dim=int(num_features),
        num_hidden=int(num_hidden),
        encoder_num_layers=edge_encoder_num_layers,
        attr_decoder_num_layers=attr_decoder_num_layers,
        struct_decoder_num_layers=struct_decoder_num_layers,
        feat_drop=in_drop,
        mask_rate=mask_rate2,
        replace_rate=replace_rate,
        drop_edge_rate=drop_edge_rate2,
        drop_path_rate=drop_path_rate2,
        predict_all_edge=predict_all_edge2,
        drop_path_length=drop_path_length2,
        walks_per_node=walks_per_node2,
        select_gano_num=select_gano_num
    )

    topology_model = PreModel(
        in_dim=int(num_features),
        num_hidden=int(num_hidden),
        encoder_num_layers=subgraph_encoder_num_layers,
        attr_decoder_num_layers=attr_decoder_num_layers,
        struct_decoder_num_layers=struct_decoder_num_layers,
        feat_drop=in_drop,
        mask_rate=mask_rate3,
        replace_rate=replace_rate,
        drop_edge_rate=drop_edge_rate3,
        drop_path_rate=drop_path_rate3,
        predict_all_edge=predict_all_edge3,
        drop_path_length=drop_path_length3,
        walks_per_node=walks_per_node3,
        select_gano_num=select_gano_num
    )
    
    return attr_model,struct_model,topology_model

def build_re_model(args):
    model = ReModel(num_features=args.num_features,
                    hid_dim=args.num_hidden,
                    dropout=args.dropout_f,
                    node_encoder_num_layers=args.node_encoder_num_layers,
                    edge_encoder_num_layers=args.edge_encoder_num_layers,
                    subgraph_encoder_num_layers=args.subgraph_encoder_num_layers,
                    attr_decoder_num_layers=args.attr_decoder_num_layers,
                    struct_decoder_num_layers=args.struct_decoder_num_layers)
    return model
# -------configs starts------------
def build_args():
    parser = argparse.ArgumentParser(description="ADAGAD")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--dataset", type=str, default="inj_cora")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_epoch", type=int, default=40,
                        help="number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=-1)

    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")

    parser.add_argument("--node_encoder_num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--edge_encoder_num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--subgraph_encoder_num_layers", type=int, default=2,
                        help="number of hidden layers")    
    parser.add_argument("--attr_decoder_num_layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--struct_decoder_num_layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.1,
                        help="attention dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="prelu")

    parser.add_argument("--replace_rate", type=float, default=0.0)

    parser.add_argument("--attr_encoder", type=str, default="gat")
    parser.add_argument("--struct_encoder", type=str, default="gcn")
    parser.add_argument("--topology_encoder", type=str, default="gcn")

    parser.add_argument("--attr_decoder", type=str, default="gcn")
    parser.add_argument("--struct_decoder", type=str, default="gcn")
    parser.add_argument("--loss_fn", type=str, default="mse")
    parser.add_argument("--alpha_l", type=float, default=3, help="`pow`coefficient for `sce` loss")
    parser.add_argument("--optimizer", type=str, default="adam")
    
    #for GAD finetune    
    parser.add_argument("--model_name", type=str, default="ADANET")
    parser.add_argument("--aggr_f", type=str, default='add')
    parser.add_argument("--max_epoch_f", type=int, default=30)
    parser.add_argument("--lr_f", type=float, default=0.001, help="learning rate for evaluation")
    parser.add_argument("--alpha_f", type=float, default=0.5)
    parser.add_argument("--dropout_f", type=float, default=0.0)
    parser.add_argument("--loss_f", type=str, default='rec')
    parser.add_argument("--loss_weight_f", type=float, default=-0.0001)
    parser.add_argument("--T_f", type=float, default=1.0)

    
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--save_model", action="store_true")

    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--use_nni", action="store_true")
    parser.add_argument("--logging", action="store_true")

    parser.add_argument("--scheduler", type=int, default=1)
    parser.add_argument("--concat_hidden", action="store_true", default=False)

    # for graph classification
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--deg4feat", action="store_true", default=False, help="use node degree as input feature")
    parser.add_argument("--batch_size", type=int, default=32)

    # for graph anomaly detection
    parser.add_argument("--use_ssl", type=int, default=1)
    parser.add_argument("--use_encoder_num", type=int, default=3)

    parser.add_argument("--attention", type=int, default=2,help="-1~0 learning attention, 0~1 unlearning attention, \
        2 hard attention, 3 soft attention")


    # for attr prediction
    parser.add_argument("--mask_rate1", type=float, default=0.1)
    
    parser.add_argument("--drop_edge_rate1", type=float, default=0)
    parser.add_argument("--predict_all_node1", type=bool, default=False)
    parser.add_argument("--predict_all_edge1", type=float, default=0)

    parser.add_argument("--drop_path_rate1", type=float, default=0)
    parser.add_argument("--drop_path_length1", type=int, default=0)
    parser.add_argument("--walks_per_node1", type=int, default=1)

    # for edge prediction
    parser.add_argument("--mask_rate2", type=float, default=0)

    parser.add_argument("--drop_edge_rate2", type=float, default=0.1)
    parser.add_argument("--predict_all_node2", type=bool, default=False)
    parser.add_argument("--predict_all_edge2", type=float, default=0)

    parser.add_argument("--drop_path_rate2", type=float, default=0)
    parser.add_argument("--drop_path_length2", type=int, default=0)
    parser.add_argument("--walks_per_node2", type=int, default=1)

    # for subgraph prediction
    parser.add_argument("--mask_rate3", type=float, default=0)
    
    parser.add_argument("--drop_edge_rate3", type=float, default=0)
    parser.add_argument("--predict_all_node3", type=bool, default=False)
    parser.add_argument("--predict_all_edge3", type=float, default=0)

    parser.add_argument("--drop_path_rate3", type=float, default=0.1)
    parser.add_argument("--drop_path_length3", type=int, default=3)
    parser.add_argument("--walks_per_node3", type=int, default=3)

    parser.add_argument("--sparse_attention_weight", type=float, default=0)

    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)

    parser.add_argument("--all_encoder_layers", type=int, default=0)

    parser.add_argument("--max_pu_epoch", type=int, default=45)
    parser.add_argument("--each_pu_epoch", type=int, default=15)

    parser.add_argument("--select_gano_num", type=int, default=30,help="select smallest G_ano")
    parser.add_argument("--sano_weight", type=int, default=1)

    args = parser.parse_args()
    return args

def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        print("Best args not found")
        return args

    print("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args
# -------configs ends------------