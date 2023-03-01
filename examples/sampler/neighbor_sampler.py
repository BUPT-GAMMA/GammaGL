from gammagl.utils import mask_to_index
from gammagl.datasets import Reddit
import tensorlayerx as tlx
import argparse
from gammagl.loader.Neighbour_sampler import Neighbor_Sampler
from pyinstrument import Profiler

dataset = Reddit('../../data/reddit')
graph = dataset[0]

train_idx = mask_to_index(graph.train_mask)


def main(args):
    sample_lists = args.sample_lists.split(',')
    for i, num in enumerate(sample_lists):
        sample_lists[i] = int(num)
    train_loader = Neighbor_Sampler(edge_index=graph.edge_index.numpy(),
                                    dst_nodes=tlx.convert_to_numpy(train_idx),
                                    sample_lists=sample_lists, batch_size=args.batch_size, shuffle=True, num_workers=0)

    profiler = Profiler()
    profiler.start()
    for dst_node, adjs, all_node in train_loader:
        pass
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_lists", type=str, default="25,10", help="sample number in each layer")
    parser.add_argument("--batch_size", type=int, default=1024, help="number of target nodes")
    args = parser.parse_args()

    main(args)
