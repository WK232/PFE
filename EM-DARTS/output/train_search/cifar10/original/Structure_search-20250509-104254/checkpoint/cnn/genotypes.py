from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# Experiment 1
NAS_201 = '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|'


#local Experiment
exp2_local = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 2), 
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), 
    reduce=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 1), ('dil_conv_5x5', 3), 
            ('dil_conv_5x5', 2), ('dil_conv_5x5', 4), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))


# Experiment 2
# 97.64   84.05
exp2_1 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2),
            ('sep_conv_5x5', 3), ('sep_conv_3x3', 4), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))

# 97.57  83.78  w
exp2_2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 3),
            ('sep_conv_5x5', 2), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

# 97.59  83.83 w
exp2_3 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('se p_conv_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3),
            ('sep_conv_3x3', 2), ('skip_connect', 2), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))

# 97.67  w 84.19 w
exp2_4 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('dil_conv_3x3', 3),
            ('skip_connect', 0), ('dil_conv_3x3', 3), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

# Experiment 3
# 97.26  w
exp3_cifar10_s1 = Genotype(
    normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 1), ('dil_conv_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 1),
            ('dil_conv_3x3', 2), ('dil_conv_5x5', 2), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
# 97.61  w
exp3_cifar10_s2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('skip_connect', 2), ('sep_conv_3x3', 4), ('sep_conv_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))

# 97.56  w
exp3_cifar10_s3 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2),
            ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

# 97.68  w
exp3_cifar10_s4 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

# 76.73 w
exp3_cifar100_s1 = Genotype(
    normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('dil_conv_3x3', 3),
            ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('dil_conv_3x3', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))

# 78.53 w
exp3_cifar100_s2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 4), ('sep_conv_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

# 79.67 w
exp3_cifar100_s3 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 4), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))
# 79.65 w
exp3_cifar100_s4 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

# 97.51  w
exp3_svhn_s1 = Genotype(
    normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2), ('skip_connect', 0),
            ('dil_conv_3x3', 3), ('skip_connect', 1), ('sep_conv_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3),
            ('dil_conv_3x3', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

# 97.66 w
exp3_svhn_s2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 4), ('skip_connect', 2)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 4), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))

# 97.695 w
exp3_svhn_s3 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 4), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))

# 97.73 w
exp3_svhn_s4 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

# s6, only parameterized operations
# 97.33
darts1 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 0),
            ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 3), ('dil_conv_3x3', 1), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
# 97.36
darts2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0),
            ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3),
            ('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))
# 97.37
darts3 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 3),
            ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2),
            ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
# 97.04
darts4 = Genotype(
    normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0),
            ('dil_conv_3x3', 3), ('dil_conv_3x3', 4), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2),
            ('sep_conv_3x3', 1), ('dil_conv_5x5', 4), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
# 97.44
emdarts1 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 3),
            ('dil_conv_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1),
            ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('dil_conv_3x3', 3)], reduce_concat=range(2, 6))

# 97.35
emdarts2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('sep_conv_5x5', 4), ('dil_conv_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2),
            ('sep_conv_3x3', 0), ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))

# 97.49
emdarts3 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

# 97.40
emdarts4 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0),
            ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

# 96.9
random1 = Genotype(
    normal=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2),
            ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('dil_conv_5x5', 3), ('dil_conv_5x5', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
# 97.09
random2 = Genotype(
    normal=[('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0),
            ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2)], normal_concat=range(2, 6),
    reduce=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 2), ('dil_conv_3x3', 2), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))
#  96.85
random3 = Genotype(
    normal=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 2), ('dil_conv_5x5', 0),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 4), ('sep_conv_5x5', 1)], normal_concat=range(2, 6),
    reduce=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 1), ('sep_conv_3x3', 4), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
# 97.11
random4 = Genotype(
    normal=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 3),
            ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

ss = Genotype(
    normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3),
            ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3),
            ('sep_conv_5x5', 2), ('sep_conv_5x5', 4), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
