import itertools

import numpy as np
import matplotlib.pyplot as plt

EPS = 1.0


def f(x):
    return np.exp(-np.abs(x) * EPS) * (EPS / 2.0)


def F(x):
    if x < 0.0:
        return 0.5 * np.exp(EPS * x)
    else:
        return 1.0 - 0.5 * np.exp(-EPS * x)


def prob_for_output_given_x0(o, x0, data_prob):
    prob_sum = 0

    for x1 in [0, 1]:
        prob_sum += data_prob[(x0, x1)] / (data_prob[(x0, 0)] + data_prob[(x0, 1)]) * f(o - x0 - x1)

    return prob_sum


def associative_privacy_budget(o, data_prob):
    prob_o_00 = prob_for_output_given_x0(o, 0, data_prob)

    prob_o_01 = prob_for_output_given_x0(o, 1, data_prob)

    return np.maximum(np.log(prob_o_00/prob_o_01), np.log(prob_o_01/prob_o_00))


def get_correlated_data(pearson_correlation_coefficient):
    data_prob = {}
    data_prob[(0, 0)] = (pearson_correlation_coefficient + 1) / 4
    data_prob[(1, 1)] = (pearson_correlation_coefficient + 1) / 4
    data_prob[(0, 1)] = 0.5 - data_prob[(0, 0)]
    data_prob[(1, 0)] = 0.5 - data_prob[(0, 0)]

    return data_prob


def get_uncorrelated_data(prob_x0, prob_x1):
    data_prob = {}

    for x0 in [0, 1]:
        for x1 in [0, 1]:
            data_prob[(x0, x1)] = prob_x0[x0] * prob_x1[x1]

    return data_prob


def generate_uncorrelated_data(n, seed=123):
    rng = np.random.default_rng(seed=seed)

    marginals = []
    for i in range(n):
        p0 = rng.uniform(0.1, 0.9)
        p1 = 1 - p0
        marginals.append({0: p0, 1: p1})

    data_prob = {}

    lists = [[0, 1] for _ in range(n)]
    for a in itertools.product(*lists):
        p = 1.0
        for i, marginal in enumerate(marginals):
            p *= marginal[a[i]]
        data_prob[a] = p

    return data_prob


if __name__ == '__main__':
    # correlated case

    data_prob = get_correlated_data(pearson_correlation_coefficient=1.0)
    print(data_prob)

    # probability that the output is less than max_output

    max_output = 0

    print(associative_privacy_budget(max_output, data_prob))

    os = np.linspace(-5, 10, 100)
    probs_given_x0_0 = [prob_for_output_given_x0(o, 0, data_prob) for o in os]
    probs_given_x0_1 = [prob_for_output_given_x0(o, 1, data_prob) for o in os]
    apgs = [associative_privacy_budget(o, data_prob) for o in os]

    plt.plot(os, probs_given_x0_0, label='Pr[O < x | x0 = 0]')
    plt.plot(os, probs_given_x0_1, label='Pr[O < x | x0 = 1]')
    plt.plot(os, apgs, label='associative epsilon')
    plt.plot(os, 100*[EPS], label='DP epsilon')

    plt.legend()
    plt.show()


    ps = np.linspace(-1, 1, 100)
    apgs_pearson_1 = [associative_privacy_budget(-50, get_correlated_data(p)) / EPS for p in ps]
    EPS = 3.0
    apgs_pearson_3 = [associative_privacy_budget(-50, get_correlated_data(p)) / EPS for p in ps]
    EPS = 10.0
    apgs_pearson_10 = [associative_privacy_budget(-50, get_correlated_data(p)) / EPS for p in ps]
    #plt.plot(ps, 100*[EPS], label='DP epsilon')
    plt.plot(ps, apgs_pearson_1, label='associative privacy loss / epsilon (epsilon = 1)')
    plt.plot(ps, apgs_pearson_3, label='associative privacy loss / epsilon (epsilon = 3)')
    plt.plot(ps, apgs_pearson_10, label='associative privacy loss / epsilon (epsilon = 10)')

    plt.legend()
    plt.show()


    # uncorrelated case

    #prob_x0 = {0: 0.1, 1: 0.9}
    #prob_x1 = {0: 0.33, 1: 0.67}

    #data_prob = get_uncorrelated_database(prob_x0, prob_x1)

    #print(associative_privacy_budget(max_output, data_prob))

