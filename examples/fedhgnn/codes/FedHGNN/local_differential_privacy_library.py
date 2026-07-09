# -*- coding: utf-8 -*-
# @Time    : 2019-05-31 12:48
# @Author  : ForestNeo
# @Email   : dr.forestneo@gmail.com
# @Software: PyCharm

#
import numpy as np
import torch


def eps2p(epsilon, n=2):
    return np.e ** epsilon / (np.e ** epsilon + n - 1)



'''矩阵形式'''
def perturbation_test(value, perturbed_value, epsilon):
    value = np.array(value)
    perturbed_value = np.array(perturbed_value)
    per_eps = epsilon#epsilon/(value.shape[0]*value.shape[1])
    rnd = np.random.random(value.shape)
    p = eps2p(per_eps)
    return np.where(rnd<p, value, np.ones((value.shape))*perturbed_value)



def k_random_response(value, values, epsilon):
    """
    the k-random response
    :param value: current value
    :param values: the possible value
    :param epsilon: privacy budget
    :return:
    """
    if not isinstance(values, list):
        raise Exception("The values should be list")
    if value not in values:
        raise Exception("Errors in k-random response")
    p = np.e ** epsilon / (np.e ** epsilon + len(values) - 1)
    if np.random.random() < p:
        return value
    values.remove(value)
    return values[np.random.randint(low=0, high=len(values))]


def k_random_response_new(item, k, epsilon):
    if not item < k:
        raise Exception("the input domain is wrong, item = %d, k = %d." % (item, k))
    p_l = 1 / (np.e ** epsilon + k - 1)
    p_h = np.e ** epsilon / (np.e ** epsilon + k - 1)
    respond_probability = np.full(shape=k, fill_value=p_l)
    respond_probability[item] = p_h
    perturbed_item = np.random.choice(a=range(k), p=respond_probability)
    return perturbed_item


def random_response_(bit_array: np.ndarray, p, q=None):
    """
    :param bit_array:
    :param p: probability of 1->1
    :param q: probability of 0->1
    update: 2020.03.06
    :return: 
    """
    q = 1-p if q is None else q
    if isinstance(bit_array, int):
        probability = p if bit_array == 1 else q
        return np.random.binomial(n=1, p=probability)
    return np.where(bit_array == 1, np.random.binomial(1, p, bit_array.shape), np.random.binomial(1, q, bit_array.shape))

'''torch version'''
def random_response(bit_array: torch.Tensor, p, q=None):
    """
    :param bit_array:
    :param p: probability of 1->1
    :param q: probability of 0->1
    update: 2020.03.06
    :return:
    """
    device = bit_array.device
    q = 1-p if q is None else q
    return torch.where(bit_array == 1, torch.distributions.bernoulli.Bernoulli(p).sample(bit_array.shape).to(device),
                       torch.distributions.bernoulli.Bernoulli(q).sample(bit_array.shape).to(device))

def unary_encoding(bit_array: np.ndarray, epsilon):
    """
    the unary encoding, the default UE is SUE
    update: 2020.02.25
    """
    if not isinstance(bit_array, np.ndarray):
        raise Exception("Type Err: ", type(bit_array))
    return symmetric_unary_encoding(bit_array, epsilon)


def symmetric_unary_encoding(bit_array: np.ndarray, epsilon):
    p = eps2p(epsilon / 2) / (eps2p(epsilon / 2) + 1)
    q = 1 / (eps2p(epsilon / 2) + 1)
    return random_response(bit_array, p, q)


def optimized_unary_encoding(bit_array: np.ndarray, epsilon):
    p = 1 / 2
    q = 1 / (eps2p(epsilon) + 1)
    return random_response(bit_array, p, q)


if __name__ == '__main__':
    # a = np.array([[0.4,-0.6],[0.7,0.2]])
    # print(discretization1(a, -1, 1))
    #
    # a = np.array([1])
    # print(discretization1(a, -1, 1))
    pass
