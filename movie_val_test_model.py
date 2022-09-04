__author__ = 'derrick.dy'
import math
import numpy as np
import time
#word2id = {}


def get_word2id(word2id_path):
    word2id = {}
    with open(word2id_path,'r',encoding='gbk') as f:
        for line in f:
            w,id = line.strip().split(' ')
            word2id[w] = int(id)
    return word2id

#label_dict = {}
#reverse_label_dict = {}
#label_list = []

def get_labels(label_dict_path,word2id_path):
    label_dict = {}
    reverse_label_dict = {}
    label_list = []
    word2id = get_word2id(word2id_path)
    with open(label_dict_path,'r',encoding='gbk') as f:
        for line in f:
            c_name,words = line.strip().split('/')
            ids = [word2id[w] for w in words.split(' ')]
            label_dict[c_name] = ids
            label_list.append(c_name)
          #  ids_str = ','.join([str(x) for x in ids if x!=3021])
            ids_str = ','.join([str(x) for x in ids])
            reverse_label_dict[ids_str] = c_name

    return label_dict, reverse_label_dict, label_list

def get_label_index(label_list, test_type):
#    zeroshot_labels = [[],[['alt.atheism','sci.electronics'],['sci.space','soc.religion.christian']],
 #              [['rec.autos','talk.politics.guns'],['comp.sys.mac.hardware','alt.atheism']],
  #             [['comp.sys.mac.hardware','misc.forsale'],['sci.crypt','sci.med']],
   #            [['comp.graphics','rec.sport.baseball'],['rec.sport.hockey','talk.politics.misc']],
    #           [['soc.religion.christian','sci.crypt'],['sci.electronics','misc.forsale']]]
    
#    zeroshot_labels = [[],[['sci.med','sci.space'],['talk.politics.misc']],
 #                  [['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware'],['sci.med']],
  #                 [['rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey'],['comp.graphics']]]

    zeroshot_labels = [[['0'],['1']],[['1'],['0']],[['2'],['3']],[['3'],['4']],[['4'],['3']]]
 #   print(test_type)
    z_labels = zeroshot_labels[test_type][0] + zeroshot_labels[test_type][1]
 #   print(z_labels)
    label_test = []
    for _l in label_list:
        if _l not in z_labels:
            label_test.append(_l)
#    print(label_test)
    indexs = list(range(len(label_test)))
    zip_label_index = zip(label_test, indexs)
    return dict(list(zip_label_index))


def get_prf(prf_file):
    label_prf = {}

    for line in open(prf_file):
        cols = line.strip().split('/')

        label_prf[cols[0]] = cols[1]
    return label_prf

# print(get_prf('movie_prf.txt'))