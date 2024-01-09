from text_create import *
from feature_extract import approximate_entropy_feature_extract, binary_matrix_rank_feature_extract
from feature_extract import cumulative_sums_feature_extract, dft_feature_extract, frequency_feature_extract
from feature_extract import frequency_within_block_feature_extract, linear_complexity_feature_extract
from feature_extract import longest_run_ones_in_a_block_feature_extract, maurers_universal_feature_extract
from feature_extract import non_overlapping_template_matching_feature_extract
from feature_extract import overlapping_template_matching_feature_extract
from feature_extract import random_excursion_feature_extract, random_excursion_variant_feature_extract
from feature_extract import run_extract, serial_feature_extract
import csv
import time
import multiprocessing


def divide(data, size):
    blocks = []
    text_block_size = size * 1024
    block_num = int(len(data) / text_block_size)
    for i in range(0, block_num):
        block = data[i * text_block_size: (i + 1) * text_block_size]
        blocks.append(block)
    return blocks


def random_feature_gen(folder_path, save_path, cryp_type, mod, size, start, end, delta):
    start_time = time.time()
    files = os.listdir(folder_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    num = start
    if end > len(files):
        end = len(files)
    for count in range(start, end):
        file = files[count]
        with open(os.path.join(folder_path, file), 'rb') as f:
            content = f.read()
            if len(content) < size * 64 * 1024:
                continue
            encrypt_content = encrypt_select(content, cryp_type, mod)
            blocks = divide(encrypt_content, size)
        features = []
        fre_block = []
        fre = []
        app = []
        run = []
        non = []
        for i in range(0, 64):
            single = blocks[i]
            fre_block.append(frequency_within_block_feature_extract.frequency_within_block_test(single))
            fre.append(frequency_feature_extract.frequency_test(single))
            app.append(approximate_entropy_feature_extract.approximate_entropy_test(single))
            run.append(run_extract.runs_test(single))
            non.extend(
                non_overlapping_template_matching_feature_extract.non_overlapping_template_matching_test(single))
        fre_block = sorted(fre_block, reverse=True)
        fre = sorted(fre, reverse=True)
        app = sorted(app, reverse=True)
        run = sorted(run, reverse=True)
        non = sorted(non, reverse=True)
        features.append(fre_block)
        features.append(fre)
        features.append(app)
        features.append(run)
        features.append(non[0: 64])
        features.append(non[64: 128])

        with open(os.path.join(save_path, '{}.csv'.format(num + delta)), 'w', newline='\r\n') as f:
            writer = csv.writer(f)
            writer.writerows(features)
        num += 1
        end_time = time.time()
        print("{}kb {}_{} feature genarate {}/{} finished! Time cost: {} s".format(size, cryp_type, mod, count, end,
                                                                                   end_time - start_time))


if __name__ == '__main__':
    folder_path = './dataset/raw_data/THUCNews/社会'
    crypt_types = ['AES', 'IDEA', 'TRIPLE_DES', 'SM4']
    crypt_mod = 'ECB'
    block_size = 4
    start = 0
    end = 500
    delta = 6000
    pool = multiprocessing.Pool()
    for crypt_type in crypt_types:
        save_path = './dataset/feature/{}_{}'.format(crypt_type, crypt_mod)
        pool.apply_async(random_feature_gen,
                         args=(folder_path, save_path, crypt_type, crypt_mod, block_size, start, end, delta))
        print("{}kb {}_{} feature genarate multiprocess start!".format(block_size, crypt_type, crypt_mod))
    pool.close()
    pool.join()
    print("Finished!")

    # feature_gen(folder_path, save_path, 'TRIPLE_DES', crypt_mod, block_size, 770, 900, delta)
