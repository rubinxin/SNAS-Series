import pickle
import genotypes
# from collections import namedtuple
# Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
genotype_file_name = f'./snas_sotl_darts_run2_genotype_child_list'

with open(genotype_file_name, 'rb') as file1:
    genotype_list = pickle.load(file1)