import numpy as np

word2id = {}
with open('./movie-review-5class/movie-review-0/word2id.txt', 'r', encoding='gbk') as f:
    for line in f:
        w, id = line.strip().split(' ')
        word2id[w] = int(id)

vocabulary_size = 55449
embedding_size = 300
emb = np.zeros((vocabulary_size + 1, embedding_size))
nlines = 0
with open('./movie-review-5class/movie-review-0/review-glove.txt') as f:
    for line in f:
        nlines += 1
        if nlines == 1:
            continue
        items = line.split()
        tid = int(items[0])
        if tid > vocabulary_size:
            print(tid)
            continue
        vec = np.array([float(t) for t in items[1:]])
        emb[tid, :] = vec
        if nlines % 20000 == 0:
            print("load {0} vectors...".format(nlines))

cnt = np.zeros(20)
for i in range(55449):
    print(i)
    for j in range(55449):
        if np.linalg.norm(emb[i]) * np.linalg.norm(emb[j]) == 0:
            similarity = -1
        else:
            similarity = np.dot(emb[i], emb[j]) / (
                    np.linalg.norm(emb[i]) * (np.linalg.norm(emb[j])))
        for k in range(20):
            if similarity>-1+k*0.1 and similarity <=-0.9+k*0.1:
                cnt[k] += 1
                break

print(cnt)

# label_query = {'0': '308,5105,984,5915',
#                '2': '3813,1745,15444,14894,16252',
#                '4': '2375,2697,1660,43967',
#                '3': '265,2375,9360,3238,5299',
#                '1': '308,6799,2986,4065,984'}
#
# prf = []
#
# label = ['0','1','2','3','4']
# for item in label:
#     index = 0
#     cnt = 0
#     tmp = []
#     tmp.append(label_query[item])
#     y_n_label = []
#
#     y_n_path = "../dataset/movie-review-5class/movie-review-"+item+"/movie-test-realscore.txt"
#     with open(y_n_path, 'r') as f:
#         for line in f:
#             y_n_label.append(int(line.strip()))
#         f.close()
#
#     for line in open('../dataset/movie-review-5class/movie-review-'+item+'/movie-test.txt'):
#         cols = line.strip().split('\t')
#         if y_n_label[index]==1 and cols[0] == label_query[item]:
#             tmp.append(cols[1])
#             cnt += 1
#         index += 1
#
#         if cnt == 3:
#             break
#
#     str = '/'.join(tmp)
#
#     prf.append(str)
#
#
# fileObject = open('movie_prf.txt', 'w')
# for ip in prf:
# 	fileObject.write(ip)
# 	fileObject.write('\n')
# fileObject.close()