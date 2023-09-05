import math
import os
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

# from scipy.special import gamma, gammainc, gammaincc
from gamma_functions import *


def lgamma(x):
    return math.log(gamma(x))


def Pr(u, eta):
    if (u == 0):
        p = math.exp(-eta)
    else:
        sum = 0.0
        for l in range(1, u + 1):
            sum += math.exp(
                -eta - u * math.log(2) + l * math.log(eta) - lgamma(l + 1) + lgamma(u) - lgamma(l) - lgamma(u - l + 1))
        p = sum
    return p


# 若需用于小于1,000,000bit的输入，需修改N和M
def overlapping_template_matching_test(bits, blen=6):
    bits = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))
    bits = list(bits)

    n = len(bits)

    m = 9
    # Build the template B as a random list of bits
    B = [1 for x in range(m)]
    # 992
    N = 92
    K = 5
    # 1032
    M = 328
    if len(bits) < (M * N):
        print("Insufficient data. %d bit provided. 1,028,016 bits required" % len(bits))
        return False, 0.0, None

    blocks = list()  # Split into N blocks of M bits
    for i in range(N):
        blocks.append(bits[i * M:(i + 1) * M])

    # Count the distribution of matches of the template across blocks: Vj
    v = [0 for x in range(K + 1)]
    for block in blocks:
        count = 0
        for position in range(M - m):
            if block[position:position + m] == B:
                count += 1

        if count >= (K):
            v[K] += 1
        else:
            v[count] += 1

    # lamd = float(M-m+1)/float(2**m) # Compute lambda and nu
    # nu = lamd/2.0

    chisq = 0.0  # Compute Chi-Square
    # pi = [0.324652,0.182617,0.142670,0.106645,0.077147,0.166269] # From spec
    pi = [0.364091, 0.185659, 0.139381, 0.100571, 0.0704323, 0.139865]  # From STS
    piqty = [int(x * N) for x in pi]

    lambd = (M - m + 1.0) / (2.0 ** m)
    eta = lambd / 2.0
    sum = 0.0
    for i in range(K):  # Compute Probabilities
        pi[i] = Pr(i, eta)
        sum += pi[i]

    pi[K] = 1 - sum

    # for block in blocks:
    #    count = 0
    #    for j in xrange(M-m+1):
    #        if B == block[j:j+m]:
    #            count += 1
    #    if ( count <= 4 ):
    #        v[count]+= 1
    #    else:
    #        v[K]+=1

    sum = 0
    chisq = 0.0
    for i in range(K + 1):
        chisq += ((v[i] - (N * pi[i])) ** 2) / (N * pi[i])
        sum += v[i]

    p = gammaincc(5.0 / 2.0, chisq / 2.0)  # Compute P value

    success = (p >= 0.01)
    return p


def overlapping_template_matching_analysis(dir, block_num, save_path):
    file_count = 0

    for root, dirs, files in os.walk(dir):
        for file in files:
            file_count += 1

    q_values = []

    for i in range(0, block_num):
        file_path = dir + '/chunk_{}.bin'.format(i + 1)
        with open(file_path, "rb") as file:
            content = file.read()
        q_value = overlapping_template_matching_test(content)
        q_values.append(round(q_value, 2))
    counter = Counter(q_values)
    fig, ax = plt.subplots(figsize=(8, 6))
    x = list(counter.keys())
    y = list(counter.values())
    ax.bar(x, y, width=0.01, edgecolor='black')
    save_path = save_path + '/overlapping_template_matching.png'
    plt.savefig(save_path)


if __name__ == "__main__":
    file_path = '../dataset/ciphertext/4kb/AES_ECB/chunk_266.bin'
    with open(file_path, "rb") as file:
        content = file.read()
    bits = content[0:128000]

    s2 = overlapping_template_matching_test(bits)
    print("p value is %s" % s2)