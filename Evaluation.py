# -*- coding: utf-8 -*-
import numpy as np


def eval_map(y_true, y_pred, rel_threshold=0):
    s = 0.
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    c = zip(y_true, y_pred)
    c = sorted(c, key=lambda x: x[1], reverse=True)
    ipos = 0
    for j, (g, p) in enumerate(c):
        if g > rel_threshold:
            ipos += 1.
            s += ipos / (j + 1.)

    if ipos == 0:
        s = 0.
    else:
        s /= float(ipos)
    return s


def precision10(y_true, y_pred, k=10, rel_threshold=0.):

    if k <= 0:
        return 0.
    s = 0.
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    c = zip(y_true, y_pred)
    #   random.shuffle(c)
    c = sorted(c, key=lambda x: x[1], reverse=True)
    ipos = 0
    precision = 0.
    for i, (g, p) in enumerate(c):
        if i >= k:
            break
        if g > rel_threshold:
            precision += 1
    precision /= float(k)
    return precision


def precision20(y_true, y_pred, k=20, rel_threshold=0.):
    if k <= 0:
        return 0.
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    c = zip(y_true, y_pred)
    c = sorted(c, key=lambda x: x[1], reverse=True)

    precision = 0.
    for i, (g, p) in enumerate(c):
        if i >= k:
            break
        if g > rel_threshold:
            precision += 1
    precision /= float(k)
    return precision


def precision50(y_true, y_pred, k=50, rel_threshold=0.):
    if k <= 0:
        return 0.
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    c = zip(y_true, y_pred)
    c = sorted(c, key=lambda x: x[1], reverse=True)
    precision = 0.
    for i, (g, p) in enumerate(c):
        if i >= k:
            break
        if g > rel_threshold:
            precision += 1
    precision /= float(k)
    return precision


def cal_map_pr(real_score_path, result_path, word2id_path, label_dict_path, test_path, label):
    real_label = []
    with open(real_score_path, 'r') as f:
        for line in f:
            real_label.append(int(line.strip()))
        f.close()

    pred_score = []
    with open(result_path, 'r') as f:
        for line in f:
            pred_score.append(float(line.strip()))
        f.close()

    word2id = {}
    with open(word2id_path, 'r') as f:
        for line in f:
            w, id = line.strip().split(' ')
            word2id[w] = int(id)
        f.close()

    label_dict = {}
    reverse_label_dict = {}
    label_list = []
    with open(label_dict_path, 'r') as f:
        for line in f:
            c_name, words = line.strip().split('/')
            ids = [word2id[w] for w in words.split(' ')]
            label_dict[c_name] = ids
            label_list.append(c_name)
            ids_str = ','.join([str(x) for x in ids])
            reverse_label_dict[ids_str] = c_name
        f.close()


    test_corpus = []
    with open(test_path, 'r') as f:
        for line in f:
            q_ids, d_ids = line.strip().split('\t')
            c_name = reverse_label_dict[q_ids]
            test_corpus.append(c_name)
        f.close()

    test_corpus = test_corpus[0: len(pred_score)]
    real_label = real_label[0: len(pred_score)]


    label_length = {}
    for _, l in enumerate(label_list):
        num = 0
        for i, j in enumerate(test_corpus):
            if (j == l) & (int(real_label[i]) == 1):
                num += 1
            else:
                continue
        label_length[l] = num

    real_dict = {}
    pred_dict = {}
    for _i, l in enumerate(label_list):
        real_dict[l] = []
        pred_dict[l] = []

    for m, n in enumerate(test_corpus):
        # print(m, n)
        real_dict[n].append(real_label[m])
        pred_dict[n].append(pred_score[m])
    # print(real_dict['0'], "\n", pred_dict['0'])
    print("# ***************** test label    ",  label, "********************")
    ap0 = eval_map(y_true=real_dict['0'], y_pred=pred_dict['0'])
    ap1 = eval_map(y_true=real_dict['1'], y_pred=pred_dict['1'])
    ap2 = eval_map(y_true=real_dict['2'], y_pred=pred_dict['2'])
    ap3 = eval_map(y_true=real_dict['3'], y_pred=pred_dict['3'])
    ap4 = eval_map(y_true=real_dict['4'], y_pred=pred_dict['4'])
    ap = eval_map(y_true=real_dict[label], y_pred=pred_dict[label])
    print("mean average precision")
    print(ap)
    # print(ap0 + ap1 + ap2 + ap3 + ap4)
    p1 = precision10(y_true=real_dict[label], y_pred=pred_dict[label])
    print("P@10:   ", p1)
    p2 = precision20(y_true=real_dict[label], y_pred=pred_dict[label])
    print("P@20:   ", p2)
    p5 = precision50(y_true=real_dict[label], y_pred=pred_dict[label])
    print("P@50:   ", p5)

    # # ******************precision********************

    # print("\n\n\n # ******************precision********************\n")
    # print(len(real_label), len(pred_score))
    real_label_1 = real_label[0: len(pred_score)]
    # print(len(real_label_1))
    test_corpus_1 = test_corpus[0: len(pred_score)]
    # print(len(test_corpus_1))

    real_label_precision = [real_label_1[i: i + 5] for i in range(0, len(real_label_1), 5)]
    pred_score_precision = [pred_score[i: i + 5] for i in range(0, len(pred_score), 5)]
    test_corpus_precision = [test_corpus_1[i: i + 5] for i in range(0, len(test_corpus_1), 5)]
    # print(len(real_label_precision))
    # print(len(pred_score_precision))
    # print(len(test_corpus_precision))

    pre_order = ['0', '1', '2', '3', '4']
    a_p = 0.
    for name in pre_order:
        TP = 0
        TN = 0
        for i, p in enumerate(real_label_precision):
            a = np.squeeze(p)
            b = np.squeeze(pred_score_precision[i])
            c = np.squeeze(test_corpus_precision[i])
            zip_label_score = zip(a, b, c)
            sort_lable_score = sorted(zip_label_score, key=lambda x: x[1], reverse=True)
            # print(sort_lable_score)
            if sort_lable_score[0][2] == name:
                if sort_lable_score[0][0] == 1:
                    TP += 1
                else:
                    TN += 1
        if (TP + TN) == 0:
            pr = 0
            print(name, TP, TN, '\t', pr)
            a_p += pr
        else:
            pr = TP / (TP + TN)
            print(name, TP, TN, '\t', pr)
            a_p += pr

    TP = 0
    TN = 0
    for i, p in enumerate(real_label_precision):
        a = np.squeeze(p)
        b = np.squeeze(pred_score_precision[i])
        c = np.squeeze(test_corpus_precision[i])
        zip_label_score = zip(a, b, c)
        sort_lable_score = sorted(zip_label_score, key=lambda x: x[1], reverse=True)
        # print(sort_lable_score)

        if sort_lable_score[0][0] == 1:
            TP += 1
        else:
            TN += 1
    pr = TP / (TP + TN)

if __name__ == '__main__':
    real_score_p = './movie-review-5class/movie-review-4/movie-test-realscore.txt'
    word2id_p = './movie-review-5class/movie-review-4/word2id.txt'
    label_dict_p = './movie-review-5class/movie-review-4/movie-label-dict.txt'
    test_p = './movie-review-5class/movie-review-4/movie-test.txt'
    result_path = './output/movie4/movie-epoch'
    epoch = ['5', '10', '15', '20','25', '30', '35', '40', '45', '50']
    zero_shot = ['0', '1', '2', '3', '4']
    for e in epoch:
        pred_path = result_path + e + '.txt'
        print(pred_path)
        cal_map_pr(real_score_p, pred_path, word2id_p, label_dict_p, test_p, zero_shot[4])
        print("\n\n\n")
