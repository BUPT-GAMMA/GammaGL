from gammagl.datasets import CA_GrQc


def test():
    datasets = CA_GrQc(r"C:\Users\76118\Desktop\cq_a", 50)

    datasets1 = datasets.read_edges_from_file(filename=r"C:\Users\76118\Desktop\cq_a\CA-GrQc_test.txt")

    datasets1 = datasets.read_edges_from_file(filename=r"C:\Users\76118\Desktop\cq_a\CA-GrQc_test_neg.txt")

    datasets1 = datasets.read_edges_from_file(filename=r"C:\Users\76118\Desktop\cq_a\CA-GrQc_train.txt")

    # datasets.read_edges_from_file()
