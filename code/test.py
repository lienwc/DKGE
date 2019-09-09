import torch
import time

import config


def _calc(h, t, r):
    return torch.norm(h + r - t, p=config.norm, dim=1).cpu().numpy().tolist()


def predict(batch, entity_emb, relation_emb):
    pos_hs = batch[:, 0]
    pos_rs = batch[:, 1]
    pos_ts = batch[:, 2]

    pos_hs = torch.IntTensor(pos_hs).cuda()
    pos_rs = torch.IntTensor(pos_rs).cuda()
    pos_ts = torch.IntTensor(pos_ts).cuda()

    p_score = _calc(entity_emb[pos_hs.type(torch.long)],
                    entity_emb[pos_ts.type(torch.long)],
                    relation_emb[pos_rs.type(torch.long)])

    return p_score


def test_head(golden_triple, train_set, entity_emb, relation_emb):
    head_batch = config.get_head_batch(golden_triple)
    value = predict(head_batch, entity_emb, relation_emb)
    golden_value = value[golden_triple[0]]
    # li = np.argsort(value)
    res = 1
    sub = 0
    for pos, val in enumerate(value):
        if val < golden_value:
            res += 1
            if (pos, golden_triple[1], golden_triple[2]) in train_set:
                sub += 1

    return res, res - sub


def test_tail(golden_triple, train_set, entity_emb, relation_emb):
    tail_batch = config.get_tail_batch(golden_triple)
    value = predict(tail_batch, entity_emb, relation_emb)
    golden_value = value[golden_triple[2]]
    # li = np.argsort(value)
    res = 1
    sub = 0
    for pos, val in enumerate(value):
        if val < golden_value:
            res += 1
            if (golden_triple[0], golden_triple[1], pos) in train_set:
                sub += 1

    return res, res - sub


def test_link_prediction(test_list, train_set, entity_emb, relation_emb):
    test_total = len(test_list)

    l_mr = 0
    r_mr = 0
    l_mrr = 0
    r_mrr = 0
    l_hit1 = 0
    l_hit3 = 0
    l_hit10 = 0
    r_hit1 = 0
    r_hit3 = 0
    r_hit10 = 0

    l_mr_filter = 0
    r_mr_filter = 0
    l_mrr_filter = 0
    r_mrr_filter = 0
    l_hit1_filter = 0
    l_hit3_filter = 0
    l_hit10_filter = 0
    r_hit1_filter = 0
    r_hit3_filter = 0
    r_hit10_filter = 0

    # last_time = time.time()
    for i, golden_triple in enumerate(test_list):
        # current_time = time.time()
        # print("loop time: %s" % str(current_time - last_time))
        # last_time = current_time

        # print('test ---' + str(i) + '--- triple')
        # print(i, end="\r")
        l_pos, l_filter_pos = test_head(golden_triple, train_set, entity_emb, relation_emb)
        r_pos, r_filter_pos = test_tail(golden_triple, train_set, entity_emb, relation_emb)  # position, 1-based

        # print(golden_triple, end=': ')
        # print('l_pos=' + str(l_pos), end=', ')
        # print('l_filter_pos=' + str(l_filter_pos), end=', ')
        # print('r_pos=' + str(r_pos), end=', ')
        # print('r_filter_pos=' + str(r_filter_pos), end='\n')

        l_mr += l_pos
        r_mr += r_pos
        l_mrr += 1 / l_pos
        r_mrr += 1 / r_pos

        if l_pos <= 1:
            l_hit1 += 1
        if l_pos <= 3:
            l_hit3 += 1
        if l_pos <= 10:
            l_hit10 += 1

        if r_pos <= 1:
            r_hit1 += 1
        if r_pos <= 3:
            r_hit3 += 1
        if r_pos <= 10:
            r_hit10 += 1

        ##################
        l_mr_filter += l_filter_pos
        r_mr_filter += r_filter_pos
        l_mrr_filter += 1 / l_filter_pos
        r_mrr_filter += 1 / r_filter_pos

        if l_filter_pos <= 1:
            l_hit1_filter += 1
        if l_filter_pos <= 3:
            l_hit3_filter += 1
        if l_filter_pos <= 10:
            l_hit10_filter += 1

        if r_filter_pos <= 1:
            r_hit1_filter += 1
        if r_filter_pos <= 3:
            r_hit3_filter += 1
        if r_filter_pos <= 10:
            r_hit10_filter += 1

    l_mr /= test_total
    r_mr /= test_total
    l_mrr /= test_total
    r_mrr /= test_total
    l_hit1 /= test_total
    l_hit3 /= test_total
    l_hit10 /= test_total
    r_hit1 /= test_total
    r_hit3 /= test_total
    r_hit10 /= test_total

    l_mr_filter /= test_total
    r_mr_filter /= test_total
    l_mrr_filter /= test_total
    r_mrr_filter /= test_total
    l_hit1_filter /= test_total
    l_hit3_filter /= test_total
    l_hit10_filter /= test_total
    r_hit1_filter /= test_total
    r_hit3_filter /= test_total
    r_hit10_filter /= test_total

    print('\t\t\tMRR\t\t\tmean_rank\t\t\thit@10\t\t\thit@3\t\t\thit@1')
    print('head(raw)\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f' % (l_mrr, l_mr, l_hit10, l_hit3, l_hit1))
    print('tail(raw)\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f' % (r_mrr, r_mr, r_hit10, r_hit3, r_hit1))
    print('average(raw)\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f' % ((l_mrr + r_mrr) / 2, (l_mr + r_mr) / 2,
                                                                              (l_hit10 + r_hit10) / 2,
                                                                              (l_hit3 + r_hit3) / 2,
                                                                              (l_hit1 + r_hit1) / 2))

    print('head(filter)\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f' % (l_mrr_filter, l_mr_filter, l_hit10_filter,
                                                                              l_hit3_filter, l_hit1_filter))
    print('tail(filter)\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f' % (r_mrr_filter, r_mr_filter, r_hit10_filter,
                                                                              r_hit3_filter, r_hit1_filter))
    print('average(filter)\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f' % ((l_mrr_filter + r_mrr_filter) / 2,
                                                                                 (l_mr_filter + r_mr_filter) / 2,
                                                                                 (l_hit10_filter + r_hit10_filter) / 2,
                                                                                 (l_hit3_filter + r_hit3_filter) / 2,
                                                                                 (l_hit1_filter + r_hit1_filter) / 2))
