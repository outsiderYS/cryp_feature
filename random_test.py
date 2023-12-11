import os

from matplotlib import pyplot as plt

from feature_extract import approximate_entropy_feature_extract, binary_matrix_rank_feature_extract
from feature_extract import cumulative_sums_feature_extract, dft_feature_extract, frequency_feature_extract
from feature_extract import frequency_within_block_feature_extract, linear_complexity_feature_extract
from feature_extract import longest_run_ones_in_a_block_feature_extract, maurers_universal_feature_extract
from feature_extract import non_overlapping_template_matching_feature_extract
from feature_extract import overlapping_template_matching_feature_extract
from feature_extract import random_excursion_feature_extract, random_excursion_variant_feature_extract
from feature_extract import run_extract, serial_feature_extract

import random
import time
from collections import Counter
import multiprocessing


random_test_dic = {0: 'approximate entropy test', 1: 'binary matrix rank test', 2: 'cumulative sums test',
                   3: 'dft test', 4: 'frequency test', 5: 'frequency within block test', 6: 'linear complexity test',
                   7: 'longest run ones in a block', 8: 'maurer\'s universal',
                   9: 'non-overlapping template matching test', 10: 'overlapping template matching test',
                   11: 'random excursion test', 12: 'random excursion variant', 13: 'run test', 14: 'serial test'}
map_position_dic = {0: []}


def select_random_test(content, select):
    if select == 0:
        q_value = approximate_entropy_feature_extract.approximate_entropy_test(content)
    if select == 1:
        q_value = binary_matrix_rank_feature_extract.binary_matrix_rank_test(content, 16, 16)
    if select == 2:
        q_value = cumulative_sums_feature_extract.cumulative_sums_test(content)
    if select == 3:
        q_value = dft_feature_extract.dft_test(content)
    if select == 4:
        q_value = frequency_feature_extract.frequency_test(content)
    if select == 5:
        q_value = frequency_within_block_feature_extract.frequency_within_block_test(content)
    if select == 6:
        q_value = linear_complexity_feature_extract.linear_complexity_test(content, 256)
    if select == 7:
        q_value = longest_run_ones_in_a_block_feature_extract.longest_run_ones_in_a_block_test(content)
    if select == 8:
        q_value = maurers_universal_feature_extract.maurers_universal_test(content, 2, 4)
    if select == 9:
        q_value = non_overlapping_template_matching_feature_extract.non_overlapping_template_matching_test(content,
                                                                                                           0)
    if select == 10:
        q_value = overlapping_template_matching_feature_extract.overlapping_template_matching_test(content, 6)
    if select == 11:
        q_value = random_excursion_feature_extract.random_excursion_test(content)
    if select == 12:
        q_value = random_excursion_variant_feature_extract.random_excursion_variant_test(content)
    if select == 13:
        q_value = run_extract.runs_test(content)
    if select == 14:
        q_value = serial_feature_extract.serial_test(content)
    return q_value


def select_analysis(file_dir, block_num, save_path, select):
    start_time = time.time()
    file_count = 0

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            file_count += 1

    q_values = []

    for i in range(0, block_num):
        file_path = file_dir + '/chunk_{}.bin'.format(i + 1)
        with open(file_path, "rb") as file:
            content = file.read()
        q_value = select_random_test(content, select)
        if isinstance(q_value, list):
            for j in q_value:
                q_values.append(round(j, 2))
        else:
            q_values.append(round(q_value, 2))
    counter = Counter(q_values)
    fig, ax = plt.subplots(figsize=(8, 6))
    x = list(counter.keys())
    y = list(counter.values())
    ax.bar(x, y, width=0.01, edgecolor='black')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel('p-value')
    ax.set_ylabel('The number of block')
    plt.title(u"{}".format(random_test_dic[select]), fontsize=14, color='k')
    save_path = save_path + '/{}.png'.format(random_test_dic[select])
    plt.savefig(save_path)
    end_time = time.time()
    print("{} blocks {} analysis finished! Time cost: {} s".format(block_num, random_test_dic[select],
                                                                 end_time - start_time))


def analysis(cipher_mod, block_size, block_num):
    dir_path = './dataset/ciphertext/{}kb/'.format(block_size) + cipher_mod
    save_path = './analysis/{}kb/'.format(block_size) + cipher_mod
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(15):
        select_analysis(dir_path, block_num, save_path, i)
    print("{} {}kb {} blocks analysis finished!".format(cipher_mod, block_size, block_num))


def compare_analysis(random_test, block_size, block_num):
    cipher_mods = ['AES_ECB', 'IDEA_ECB', 'RSA', 'TRIPLE_DES_ECB', 'SM2', 'SM4_ECB']
    save_path = './compare_analysis/{}kb/'.format(block_size)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig = plt.figure(figsize=(24, 12), dpi=300)

    start_time = time.time()

    for num in range(len(cipher_mods)):
        cipher_mod = cipher_mods[num]
        dir_path = './dataset/ciphertext/{}kb/'.format(block_size) + cipher_mod

        q_values = []
        for i in range(0, block_num):
            file_path = dir_path + '/chunk_{}.bin'.format(i + 1)
            with open(file_path, "rb") as file:
                content = file.read()
            q_value = select_random_test(content, random_test)
            if isinstance(q_value, list):
                for j in q_value:
                    q_values.append(round(j, 2))
            else:
                q_values.append(round(q_value, 2))

        counter = Counter(q_values)
        ax = plt.subplot(2, 3, num + 1)
        x = list(counter.keys())
        y = list(counter.values())
        ax.bar(x, y, width=0.01, edgecolor='black')
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlabel('p-value')
        ax.set_ylabel('The number of block')
        ax.set_title(u"{}".format(cipher_mod))

    fig.suptitle('{} compare'.format(random_test_dic[random_test]), fontsize=28, color='k')
    save_path = save_path + '/{}_compare.png'.format(random_test_dic[random_test])
    plt.savefig(save_path)

    end_time = time.time()
    print("{} blocks {} compare analysis finished! Time cost: {} s".format(block_num, random_test_dic[random_test],
                                                                           end_time - start_time))


if __name__ == '__main__':
    # cipher_mods = ['AES_ECB', 'IDEA_ECB', 'RSA', 'TRIPLE_DES_ECB', 'SM2', 'SM4_ECB']
    block_size = 4
    block_num = 500
    #
    # pool = multiprocessing.Pool()
    # for cipher_mod in cipher_mods:
    #     pool.apply_async(analysis, args=(cipher_mod, block_size, block_num))
    #     print("{} {}kb {} blocks analysis multiprocess start!".format(cipher_mod, block_size, block_num))
    # pool.close()
    # pool.join()
    # print("All ciphermod {}kb {} blocks analysis finished!".format(block_size, block_num))

    pool = multiprocessing.Pool()
    for i in range(15):
        pool.apply_async(compare_analysis, args=(i, block_size, block_num))
        print("{}kb {} blocks {} compare analysis multiprocess start!".format(block_size, block_num,
                                                                              random_test_dic[i]))
    pool.close()
    pool.join()
    print("All ciphermod {}kb {} blocks compare analysis finished!".format(block_size, block_num))
