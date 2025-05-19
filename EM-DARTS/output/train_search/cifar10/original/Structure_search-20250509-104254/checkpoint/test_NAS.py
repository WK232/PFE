from nas_201_api import NASBench201API as API

# Create an API without the verbose log
api = API('NAS-Bench-201-v1_1-096897.pth', verbose=False)
# The default path for benchmark file is '{:}/{:}'.format(os.environ['TORCH_HOME'], 'NAS-Bench-201-v1_1-096897.pth')

index = api.query_index_by_arch(
    '|skip_connect~0|+|skip_connect~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|nor_conv_3x3~2|')
api.show(index)
#  85.07   |skip_connect~0|+|skip_connect~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|nor_conv_3x3~2|
# 90.11 |nor_conv_3x3~0|+|nor_conv_3x3~0|skip_connect~1|+|nor_conv_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|
# 90.27  |nor_conv_3x3~0|+|nor_conv_3x3~0|skip_connect~1|+|nor_conv_1x1~0|nor_conv_3x3~1|nor_conv_3x3~2|
# 89.81  |nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|
# 90.14 |nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|nor_conv_1x1~0|skip_connect~1|nor_conv_1x1~2|
# 90.10 |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_1x1~0|skip_connect~1|nor_conv_1x1~2|
# 89.60  |skip_connect~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_1x1~0|nor_conv_3x3~1|nor_conv_1x1~2|
# 89.33  |nor_conv_1x1~0|+|nor_conv_3x3~0|nor_conv_1x1~1|+|nor_conv_1x1~0|nor_conv_1x1~1|skip_connect~2|
# 89.90   |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_1x1~1|+|nor_conv_3x3~0|nor_conv_1x1~1|nor_conv_1x1~2|
# 89.61 |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_1x1~1|+|nor_conv_3x3~0|nor_conv_3x3~1|skip_connect~2|
# 90.11  |nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_1x1~1|+|nor_conv_1x1~0|nor_conv_3x3~1|skip_connect~2|
# 89.43% |nor_conv_3x3~0|+|skip_connect~0|nor_conv_1x1~1|+|nor_conv_1x1~0|skip_connect~1|nor_conv_1x1~2|
# 91.53 |nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|
# 89.77 |nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_1x1~1|+|nor_conv_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|
# 90.44 |nor_conv_3x3~0|+|skip_connect~0|nor_conv_3x3~1|+|nor_conv_1x1~0|nor_conv_3x3~1|nor_conv_3x3~2|
# 89.21 |skip_connect~0|+|skip_connect~0|nor_conv_3x3~1|+|nor_conv_1x1~0|nor_conv_1x1~1|nor_conv_1x1~2|
# 86.66 |skip_connect~0|+|skip_connect~0|nor_conv_1x1~1|+|nor_conv_1x1~0|nor_conv_1x1~1|nor_conv_1x1~2|
# 89.97 |skip_connect~0|+|nor_conv_3x3~0|nor_conv_1x1~1|+|nor_conv_1x1~0|nor_conv_3x3~1|nor_conv_1x1~2|
# 90.06 |nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_1x1~1|+|nor_conv_1x1~0|nor_conv_3x3~1|nor_conv_1x1~2|
# 90.18 |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_3x3~0|nor_conv_1x1~1|nor_conv_3x3~2|
# 89.15 |skip_connect~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_1x1~0|nor_conv_1x1~1|skip_connect~2|
# 89.98 |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_3x3~0|nor_conv_3x3~1|nor_conv_1x1~2|
# 89.61 |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_1x1~1|+|nor_conv_3x3~0|nor_conv_3x3~1|skip_connect~2|
# 88.96 |nor_conv_1x1~0|+|nor_conv_3x3~0|nor_conv_1x1~1|+|nor_conv_3x3~0|nor_conv_3x3~1|skip_connect~2|
# 89.65 |nor_conv_1x1~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|nor_conv_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|
# 89.86 |nor_conv_1x1~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|nor_conv_3x3~0|nor_conv_1x1~1|nor_conv_3x3~2|
# 89.40 |nor_conv_1x1~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|nor_conv_3x3~0|nor_conv_1x1~1|skip_connect~2|
# 86.54 |skip_connect~0|+|nor_conv_1x1~0|nor_conv_1x1~1|+|nor_conv_1x1~0|nor_conv_1x1~1|nor_conv_1x1~2|
# 90.32 |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_1x1~0|nor_conv_3x3~1|nor_conv_1x1~2|
# 89.56 |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|
# 90.44 |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_1x1~0|skip_connect~1|nor_conv_3x3~2|
# 90.19 |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_3x3~0|skip_connect~1|nor_conv_3x3~2|
# 90.20  |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|
# 89.98 |nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|nor_conv_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|
#  86.55 |nor_conv_1x1~0|+|skip_connect~0|nor_conv_1x1~1|+|skip_connect~0|nor_conv_1x1~1|skip_connect~2|
# 90.37 |nor_conv_1x1~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_1x1~1|nor_conv_1x1~2|
# 91.42 |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_1x1~1|nor_conv_1x1~2|
# 91.34 |nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_1x1~1|nor_conv_1x1~2|
# 90.79 |nor_conv_1x1~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_1x1~1|nor_conv_1x1~2|
# 90.40 |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_1x1~0|nor_conv_1x1~1|nor_conv_3x3~2|
# 85.07 |nor_conv_3x3~0|+|skip_connect~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|skip_connect~2|
# 85.13 |skip_connect~0|+|skip_connect~0|nor_conv_1x1~1|+|skip_connect~0|skip_connect~1|nor_conv_1x1~2|`
# 91.12 |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|nor_conv_3x3~2|
# 90.86 |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|skip_connect~2|
# 73.55 |skip_connect~0|+|skip_connect~0|nor_conv_1x1~1|+|skip_connect~0|skip_connect~1|skip_connect~2|
# 90.59 |skip_connect~0|+|skip_connect~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|
# 87.30 |nor_conv_3x3~0|+|skip_connect~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|
# 39.77 |skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|
# 75.21 |skip_connect~0|+|skip_connect~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|skip_connect~2|
# 90.57 |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_1x1~0|nor_conv_3x3~1|nor_conv_3x3~2|
# 91.21  |nor_conv_3x3~0|+|skip_connect~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|
# 87.00 |nor_conv_3x3~0|+|skip_connect~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|skip_connect~2|
# 90.51 |nor_conv_3x3~0|+|skip_connect~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|nor_conv_3x3~2|
# 90.88 |nor_conv_1x1~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|nor_conv_1x1~2|
# 90.24 |nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|nor_conv_1x1~0|nor_conv_3x3~1|nor_conv_3x3~2|
# |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_1x1~0|nor_conv_1x1~1|nor_conv_1x1~2| 90.12% 93.50%
# |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|nor_conv_1x1~2|  91.26%  93.89%
# |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2| #rool   91.55% 94.36%  71.17% 71.27%  46.03% 46.33%
# |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_1x1~1|nor_conv_3x3~2| 91.50%  94.37%


# |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_1x1~2| 最好的结构  91.61% 94.37%  73.22% 46.71%
