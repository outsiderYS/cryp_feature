import random
from PIL import Image
import numpy as np
import multiprocessing
from feature_extract import frequency_within_block_feature_extract, frequency_feature_extract, \
    approximate_entropy_feature_extract, run_extract, non_overlapping_template_matching_feature_extract
from text_create import encrypt_select


def feature_extract(data, alg, num, istrain):
    step = 4 * 1024
    fre_block = []
    fre = []
    app = []
    run = []
    non1 = []
    non2 = []
    for i in range(0, 64):
        single = data[i * step: (i + 1) * step]
        fre_block.append(frequency_within_block_feature_extract.frequency_within_block_test(single))
        fre.append(frequency_feature_extract.frequency_test(single))
        app.append(approximate_entropy_feature_extract.approximate_entropy_test(single))
        run.append(run_extract.runs_test(single))
        non_total = non_overlapping_template_matching_feature_extract.non_overlapping_template_matching_test(single)
        non1.append(non_total[0])
        non2.append(non_total[1])
    feature = [fre_block, fre, app, run, non1, non2]
    feature = np.array(feature) * 255
    gray_image = np.uint8(feature)
    image = Image.fromarray(gray_image, 'L')
    if istrain:
        image.save('./random_dataset/train/{}/{}.png'.format(alg, num))
    else:
        image.save('./random_dataset/test/{}/{}.png'.format(alg, num))


def image_gen(data, alg, num, istrain):
    encrypt_content = encrypt_select(data, alg, 'ECB')
    feature_extract(encrypt_content, alg, num, istrain)


if __name__ == '__main__':
    size_kb = 4 * 64

    train_num = 8000
    test_num = 2000

    for i in range(0, train_num):
        generated_content = bytes([random.randint(0, 255) for _ in range(size_kb * 1024)])
        alg = ['AES', 'IDEA', 'TRIPLE_DES', 'RSA', 'SM2', 'SM4']

        p_aes = multiprocessing.Process(target=image_gen, args=(generated_content, 'AES', 0, True))
        p_idea = multiprocessing.Process(target=image_gen, args=(generated_content, 'IDEA', 0, True))
        p_3des = multiprocessing.Process(target=image_gen, args=(generated_content, 'TRIPLE_DES', 0, True))
        p_rsa = multiprocessing.Process(target=image_gen, args=(generated_content, 'RSA', 0, True))
        p_sm2 = multiprocessing.Process(target=image_gen, args=(generated_content, 'SM2', 0, True))
        p_sm4 = multiprocessing.Process(target=image_gen, args=(generated_content, 'SM4', 0, True))

        p_aes.start()
        p_idea.start()
        p_3des.start()
        p_rsa.start()
        p_sm2.start()
        p_sm4.start()

        p_aes.join()
        p_idea.join()
        p_3des.join()
        p_rsa.join()
        p_sm2.join()
        p_sm4.join()

        print("trainSet:{}/{}\n".format(i, train_num))

    for i in range(0, test_num):
        generated_content = bytes([random.randint(0, 255) for _ in range(size_kb * 1024)])
        alg = ['AES', 'IDEA', 'TRIPLE_DES', 'RSA', 'SM2', 'SM4']

        p_aes = multiprocessing.Process(target=image_gen, args=(generated_content, 'AES', 0, False))
        p_idea = multiprocessing.Process(target=image_gen, args=(generated_content, 'IDEA', 0, False))
        p_3des = multiprocessing.Process(target=image_gen, args=(generated_content, 'TRIPLE_DES', 0, False))
        p_rsa = multiprocessing.Process(target=image_gen, args=(generated_content, 'RSA', 0, False))
        p_sm2 = multiprocessing.Process(target=image_gen, args=(generated_content, 'SM2', 0, False))
        p_sm4 = multiprocessing.Process(target=image_gen, args=(generated_content, 'SM4', 0, False))

        p_aes.start()
        p_idea.start()
        p_3des.start()
        p_rsa.start()
        p_sm2.start()
        p_sm4.start()

        p_aes.join()
        p_idea.join()
        p_3des.join()
        p_rsa.join()
        p_sm2.join()
        p_sm4.join()

        print("testSet:{}/{}\n".format(i, test_num))
