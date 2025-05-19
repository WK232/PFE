from cnn import *
spaces = spaces_dict['s5']
for i in range(4):
    model = Network(16, 10, 8, spaces, 'cifar10')

    print(f'genotype = {model.genotype()}')
