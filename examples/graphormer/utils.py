def decrease_to_max_value(x, max_value):
    """
    将输入张量x中大于指定最大值max_value的元素替换为max_value。

    :param x: 输入张量
    :param max_value: 指定的最大值
    :return: 替换后的张量
    """
    x[x > max_value] = max_value
    return x
