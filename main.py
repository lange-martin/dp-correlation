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


def prob_for_output_given_x0(o, x0, data_prob, n=2):
    prob_sum = 0
    prob_sum_value_x0 = 0

    lists = [[0, 1] for _ in range(n-1)]
    for a in itertools.product(*lists):
        x0a = (x0,) + a

        tuple_sum = 0
        for i in range(n):
            tuple_sum += x0a[i]

        prob_sum_value_x0 += data_prob[x0a]
        prob_sum += data_prob[x0a] * F(o - tuple_sum)

    prob_sum = prob_sum / prob_sum_value_x0

    return prob_sum


def associative_privacy_loss(o, data_prob, n=2):
    prob_o_00 = prob_for_output_given_x0(o, 0, data_prob, n)

    prob_o_01 = prob_for_output_given_x0(o, 1, data_prob, n)

    return np.maximum(np.log(prob_o_00/prob_o_01), np.log(prob_o_01/prob_o_00))


def privacy_loss(o, n=2):
    # calculate maximum privacy loss over all possibilities of the remaining dataset (x1)
    privacy_loss = 0.0

    for x1 in [0, 1]:
        prob_o_0 = f(o - 0 - x1)
        prob_o_1 = f(o - 1 - x1)
        privacy_loss_t = np.maximum(np.log(prob_o_0/prob_o_1), np.log(prob_o_1/prob_o_0))
        privacy_loss = max(privacy_loss, privacy_loss_t)

    return privacy_loss


def get_correlated_data(pearson_correlation_coefficient):
    data_prob = {}
    data_prob[(0, 0)] = (pearson_correlation_coefficient + 1) / 4
    data_prob[(1, 1)] = (pearson_correlation_coefficient + 1) / 4
    data_prob[(0, 1)] = 0.5 - data_prob[(0, 0)]
    data_prob[(1, 0)] = 0.5 - data_prob[(0, 0)]

    return data_prob


def get_multiple_correlated_rvs(pearson, n):
    data_prob = get_correlated_data(pearson)
    data_prob_prev = data_prob
    data_prob_two = get_correlated_data(pearson)

    for i in range(2, n):
        data_prob = {}

        # sum up probability of last rv being 0 (1)
        sum_prob_xn = {0: 0.0, 1: 0.0}
        for xn in [0, 1]:
            lists = [[0, 1] for _ in range(i - 1)]
            for a in itertools.product(*lists):
                a = a + (xn,)
                sum_prob_xn[xn] += data_prob_prev[a]

        lists = [[0, 1] for _ in range(i + 1)]
        for a in itertools.product(*lists):
            data_prob[a] = (data_prob_prev[a[:-1]] / sum_prob_xn[a[-2]]) * data_prob_two[a[-2:]]

        data_prob_prev = data_prob

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


def combine_distributions(data_prob_0, n0, data_prob_1, n1):
    data_prob = {}

    lists = [[0, 1] for _ in range(n0+n1)]
    for a in itertools.product(*lists):
        data_prob[a] = data_prob_0[a[:n0]] * data_prob_1[a[n0:]]

    return data_prob


def get_mixed_dataset(pearson, n_correlated, n_independent):
    data_prob_c = get_multiple_correlated_rvs(pearson, n_correlated)
    data_prob_i = generate_uncorrelated_data(n_independent)
    return combine_distributions(data_prob_c, n_correlated, data_prob_i, n_independent)


if __name__ == '__main__':
    apl_2 = associative_privacy_loss(-50, get_mixed_dataset(0.75, 2, 0), 2)
    apl_8 = associative_privacy_loss(-50, get_mixed_dataset(0.75, 8, 0), 8)

    print(apl_2)
    print(apl_8)

    # correlated case (many variate)
    Nc = 2
    Ni = 0
    N = Nc + Ni
    Pearson = 1.0

    data_prob = get_mixed_dataset(Pearson, Nc, Ni)
    print(data_prob)

    os = np.linspace(-1, N + 1, 100)
    probs_given_x0_0 = [prob_for_output_given_x0(o, 0, data_prob, N) for o in os]
    probs_given_x0_1 = [prob_for_output_given_x0(o, 1, data_prob, N) for o in os]
    apgs = [associative_privacy_loss(o, data_prob, N) for o in os]

    plt.plot(os, probs_given_x0_0, label='Pr[O ≤ x | D_1 = 0]')
    plt.plot(os, probs_given_x0_1, label='Pr[O ≤ x | D_1 = 1]')
    plt.plot(os, apgs, label='associative privacy loss')
    plt.plot(os, 100*[EPS], label='DP epsilon')
    #plt.plot(os, [privacy_loss(o) for o in os], label='privacy loss')

    plt.title(f"Pearson: {Pearson}, Correlated Records: {Nc}, Independent Records: {Ni}")
    plt.legend()
    plt.show()


    ps = np.linspace(0, 1, 100)
    Nc = 2
    apgs_pearson_2 = [associative_privacy_loss(-50, get_mixed_dataset(p, Nc, Ni), Nc + Ni) for p in ps]
    Nc = 4
    apgs_pearson_4 = [associative_privacy_loss(-50, get_mixed_dataset(p, Nc, Ni), Nc + Ni) for p in ps]
    Nc = 8
    apgs_pearson_8 = [associative_privacy_loss(-50, get_mixed_dataset(p, Nc, Ni), Nc + Ni) for p in ps]
    #plt.plot(ps, 100*[EPS], label='DP epsilon')
    plt.plot(ps, apgs_pearson_2, label='associative privacy loss (2 correlated records)')
    plt.plot(ps, apgs_pearson_4, label='associative privacy loss (4 correlated records)')
    plt.plot(ps, apgs_pearson_8, label='associative privacy loss (8 correlated records)')

    plt.legend()
    plt.show()

    heatmap = np.zeros((100, 9))
    for i, pearson in enumerate(np.linspace(-1, 1, 100)):
        for j, n_correlated in enumerate(range(2, 11)):
            heatmap[i, j] = associative_privacy_loss(-50, get_mixed_dataset(pearson, n_correlated, 0), n_correlated)
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.show()


