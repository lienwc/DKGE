import torch

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

    return res - sub


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

    return res - sub


def validate(pht_o, pr_o):
    entity_emb = torch.Tensor(config.entity_total, config.dim)
    # entity_emb = torch.Tensor(len(pht_o), config.dim)
    for k, v in pht_o.items():
        entity_emb[int(k)] = torch.Tensor(v)

    relation_emb = torch.Tensor(config.relation_total, config.dim)
    # relation_emb = torch.Tensor(len(pr_o), config.dim)
    for k, v in pr_o.items():
        relation_emb[int(k)] = torch.Tensor(v)

    entity_emb = entity_emb.cuda()
    relation_emb = relation_emb.cuda()

    train_set = set(config.train_list)
    valid_total = len(config.valid_list)

    l_mrr_filter = 0.0
    r_mrr_filter = 0.0

    for i, golden_triple in enumerate(config.valid_list):
        # print(i, end="\r")
        l_filter_pos = test_head(golden_triple, train_set, entity_emb, relation_emb)
        r_filter_pos = test_tail(golden_triple, train_set, entity_emb, relation_emb)  # position, 1-based

        l_mrr_filter += 1 / l_filter_pos
        r_mrr_filter += 1 / r_filter_pos

    l_mrr_filter /= valid_total
    r_mrr_filter /= valid_total

    # return l_mrr_filter
    return (l_mrr_filter + r_mrr_filter) / 2


def validate2(entity_emb, relation_emb):
    train_set = set(config.train_list)
    valid_total = len(config.valid_list)

    l_mrr_filter = 0.0
    r_mrr_filter = 0.0

    for i, golden_triple in enumerate(config.valid_list):
        print(i, end="\r")
        l_filter_pos = test_head(golden_triple, train_set, entity_emb, relation_emb)
        r_filter_pos = test_tail(golden_triple, train_set, entity_emb, relation_emb)  # position, 1-based

        l_mrr_filter += 1 / l_filter_pos
        r_mrr_filter += 1 / r_filter_pos

    l_mrr_filter /= valid_total
    r_mrr_filter /= valid_total

    # return l_mrr_filter
    return (l_mrr_filter + r_mrr_filter) / 2


def transe_validate(state_dict):
    entity_emb = state_dict['entity_emb']
    relation_emb = state_dict['relation_emb']

    train_set = set(config.train_list)
    valid_total = len(config.valid_list)

    l_mrr_filter = 0.0
    r_mrr_filter = 0.0

    for i, golden_triple in enumerate(config.valid_list):
        l_filter_pos = test_head(golden_triple, train_set, entity_emb, relation_emb)
        r_filter_pos = test_tail(golden_triple, train_set, entity_emb, relation_emb)  # position, 1-based

        l_mrr_filter += 1 / l_filter_pos
        r_mrr_filter += 1 / r_filter_pos

    l_mrr_filter /= valid_total
    r_mrr_filter /= valid_total

    # return l_mrr_filter
    return (l_mrr_filter + r_mrr_filter) / 2
