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
import csv
import numpy as np
from scipy import stats

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

    start_time = time.time()

    q_value_list = []
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
        q_values.sort()
        q_value_list.append(q_values)

    with open(os.path.join(save_path, '{}_compare.csv'.format(random_test_dic[random_test])), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(q_value_list)

    end_time = time.time()
    print("{} blocks {} compare analysis finished! Time cost: {} s".format(block_num, random_test_dic[random_test],
                                                                           end_time - start_time))


def kl_divergence(list_a, list_b):
    kl_forward = stats.entropy(list_a, list_b)
    kl_backward = stats.entropy(list_b, list_a)
    kl = (kl_forward + kl_backward)/2
    return kl


def laplace_smoothing(list, nums, alpha):
    for i in range(len(list)):
        up = list[i] + alpha
        down = nums + alpha*len(list)
        list[i] = up / down
    return list


def times_trans(list):
    times_list = [0] * 101
    for i in list:
        times_list[int(float(i)*100)] += 1
    return times_list


def compare_kl():
    cipher_mods = ['AES_ECB', 'IDEA_ECB', 'RSA', 'TRIPLE_DES_ECB', 'SM2', 'SM4_ECB']
    raplace_alpha = 0.001
    save_path = "./compare_analysis/4kb/kl_compare.csv"

    csv_header = ['random test']
    for i in range(len(cipher_mods) - 1):
        for j in range(i + 1, len(cipher_mods)):
            csv_header.append("{}_{}_kl".format(cipher_mods[i], cipher_mods[j]))
    csv_header.append("kl_mean")

    csv_rows = []
    for random_test in range(0, 15):
        file_path = "./compare_analysis/4kb/{}_compare.csv".format(random_test_dic[random_test])

        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            # 遍历每一行数据
            count = 0
            p_value_list = []
            for row in reader:
                # 处理每一行数据
                if count % 2 == 0:
                    p_value_list.append(row)
                count += 1
        all_list = []
        AES_list = p_value_list[0]
        nums = len(AES_list)
        AES_list = times_trans(AES_list)
        AES_list = laplace_smoothing(AES_list, nums, raplace_alpha)
        all_list.append(AES_list)

        IDEA_list = p_value_list[1]
        IDEA_list = times_trans(IDEA_list)
        IDEA_list = laplace_smoothing(IDEA_list, nums, raplace_alpha)
        all_list.append(IDEA_list)

        RSA_list = p_value_list[2]
        RSA_list = times_trans(RSA_list)
        RSA_list = laplace_smoothing(RSA_list, nums, raplace_alpha)
        all_list.append(RSA_list)

        Triple_DES_list = p_value_list[3]
        Triple_DES_list = times_trans(Triple_DES_list)
        Triple_DES_list = laplace_smoothing(Triple_DES_list, nums, raplace_alpha)
        all_list.append(Triple_DES_list)

        SM2_list = p_value_list[4]
        SM2_list = times_trans(SM2_list)
        SM2_list = laplace_smoothing(SM2_list, nums, raplace_alpha)
        all_list.append(SM2_list)

        SM4_list = p_value_list[5]
        SM4_list = times_trans(SM4_list)
        SM4_list = laplace_smoothing(SM4_list, nums, raplace_alpha)
        all_list.append(SM4_list)

        kl_list = ["{}".format(random_test_dic[random_test])]
        for i in range(len(cipher_mods) - 1):
            for j in range(i+1, len(cipher_mods)):
                kl_i_j = kl_divergence(all_list[i], all_list[j])
                kl_list.append(kl_i_j)
        kl_list.append(np.mean(kl_list[1:]))
        csv_rows.append(kl_list)

    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)
    print("success")


if __name__ == '__main__':
    # cipher_mods = ['AES_ECB', 'IDEA_ECB', 'RSA', 'TRIPLE_DES_ECB', 'SM2', 'SM4_ECB']
    # block_size = 4
    # block_num = 2000
    # #
    # # pool = multiprocessing.Pool()
    # # for cipher_mod in cipher_mods:
    # #     pool.apply_async(analysis, args=(cipher_mod, block_size, block_num))
    # #     print("{} {}kb {} blocks analysis multiprocess start!".format(cipher_mod, block_size, block_num))
    # # pool.close()
    # # pool.join()
    # # print("All ciphermod {}kb {} blocks analysis finished!".format(block_size, block_num))
    #
    # pool = multiprocessing.Pool()
    # for i in range(15):
    #     pool.apply_async(compare_analysis, args=(i, block_size, block_num))
    #     print("{}kb {} blocks {} compare analysis multiprocess start!".format(block_size, block_num,
    #                                                                           random_test_dic[i]))
    # pool.close()
    # pool.join()
    # print("All ciphermod {}kb {} blocks compare analysis finished!".format(block_size, block_num))
    # compare_analysis(4, block_size, block_num)
    compare_kl()
