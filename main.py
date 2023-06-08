import numpy as np
import matplotlib.pyplot as plt

EPS = 4.0


def F(x, epsilon=EPS):
    if x < 0.0:
        return 0.5 * np.exp(epsilon * x)
    else:
        return 1.0 - 0.5 * np.exp(-epsilon * x)


def prob_for_output_given_x0(o, x0, data_prob):
    prob_sum = 0

    for x1 in [0, 1]:
        prob_sum += data_prob[(x0, x1)] / (data_prob[(x0, 0)] + data_prob[(x0, 1)]) * F(o - x0 - x1)

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


if __name__ == '__main__':

    # correlated case

    data_prob = get_correlated_data(pearson_correlation_coefficient=1)
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
    apgs_pearson = [associative_privacy_budget(-50, get_correlated_data(p)) for p in ps]
    plt.plot(ps, 100*[EPS], label='DP epsilon')
    plt.plot(ps, apgs_pearson, label='associative epsilon for pearson coefficient')

    plt.legend()
    plt.show()


    # uncorrelated case

    #prob_x0 = {0: 0.1, 1: 0.9}
    #prob_x1 = {0: 0.33, 1: 0.67}

    #data_prob = get_uncorrelated_database(prob_x0, prob_x1)

    #print(associative_privacy_budget(max_output, data_prob))

