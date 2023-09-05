1.基于频率检验
cryp_feature/feature_extract/frequency_feature_extract.py 
frequency_test(data) 
input:’rb’		output:p-value

2.块内频数检验  cryp_feature/feature_extract/frequency_within_block_feature_extract.py 
frequency_within_block_test(data) 
input:’rb’		output:p-value

3.游程检验 cryp_feature/feature_extract/run_extract.py 
runs_test(data) 
input:’rb’		output:p-value

4.块内最长游程检验  cryp_feature/feature_extract/longest_run_ones_in_a_block_feature_extract.py 
longest_run_ones_in_a_block_test(data) 
input:’rb’		output:p-value

5.二元矩阵秩检验 cryp_feature/feature_extract/binary_matrix_rank_feature_extract.py
binary_matrix_rank_test(bits, M=32, Q=32)
input:’rb’		output:p-value
# 若想在输入为4kb的情况下使用，需调整M和Q的大小使得M*Q*38<4096*8

6.离散傅里叶变换检验 cryp_feature/feature_extract/dft_feature_extract.py
dft_test(bits):
input:’rb’		output:p-value

7.非重叠模块匹配检验 cryp_feature/feature_extract/non_overlapping_template_matching_feature_extract.py
non_overlapping_template_matching_test(bits, temp_select=0)
# 若需用于小于1,000,000bit的输入，需修改N和M，以及选用的合适的templates，返回的为一个长度与选用的template相同的p-value列表
input:’rb’		output:list(p-value)

8.重叠模块匹配检验 cryp_feature/feature_extract/overlapping_template_matching_feature_extract.py
overlapping_template_matching_test(bits, blen=6):
# 若需用于小于1,000,000bit的输入，需修改函数内的N和M，如92和328
input:’rb’		output:p-value

9.Maurer 的通用统计检验 cryp_feature/feature_extract/maurers_universal_feature_extract.py
maurers_universal_test(bits, patternlen=None, initblocks=None):
# 比特长度至少为 387,840，若需使用4kb的输入，则需要修改patternlen和initblocks，例如2和4
input:’rb’		output:p-value

10.线性复杂度检验 cryp_feature/feature_extract/linear_complexity_feature_extract.py
linear_complexity_test(bits, patternlen=None)
#该功能需要输入大于10^6，用于4kb输入需调整patternlen大小，如256
input:’rb’		output:p-value

11.序列检验 cryp_feature/feature_extract/serial_feature_extract.py
serial_test(bits, patternlen=None)
# 需要根据输入的数据，控制patternlen,注意输出为两个p-value
input:’rb’		output:[p-value, p-value]

12.近似熵检验 cryp_feature/feature_extract/approximate_entropy_feature_extract.py
approximate_entropy_test(bits):
# 对于不同大小的输入需要调整其中的m值
input:’rb’		output:p-value

13.累加和检验 cryp_feature/feature_extract/cumulative_sums_feature_extract.py
cumulative_sums_test(bits)
# 注意输出为一个p-value list，分别为前向和后向的两个p-value
input:’rb’		output:[p-value, p-value]

14.随机游走检验 cryp_feature/feature_extract/random_excursion_feature_extract.py
random_excursion_test(bits)
# 注意输出为一个p-value list，其中包括8个p-value
input:’rb’		output:list(p-value)

15.随机游走状态频数检验 cryp_feature/feature_extract/random_excursion_variant_feature_extract.py
random_excursion_variant_test(bits)
# 注意输出为一个p-value list,其中包含18个p-value
input:’rb’		output:list(p-value)

from joblib import dump, load
# randomforest
dump(clf, "../checkpoints/rf_checkpoints.joblib") # save
clf = load("../checkpoints/rf_checkpoints.joblib") # load

import tensorflow as tf
# save model
output_path = "../checkpoints/xxx_checkpoints_epoch_150"
model.save(output_path)
# load model
cnn_model = tf.keras.models.load_model("../checkpoints/cnn_checkpoints_epoch_150") # cnn
bp_model = tf.keras.models.load_model("../checkpoints/bpnn_checkpoints_epoch_150") # bpnn
lstm = tf.keras.models.load_model("../checkpoints/lstm_checkpoints_epoch_150") # lstm
