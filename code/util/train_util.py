import random
import torch
import torch.nn as nn
import numpy as np


def read_file(file_name):
    data = []  # [(h, r, t)]
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            li = line.split()
            if len(li) == 3:
                data.append((int(li[0]), int(li[2]), int(li[1])))
    return data


def get_total(file_name):
    with open(file_name, 'r', encoding="utf-8") as f:
        return int(f.readline())


def get_basic_info(train_set):
    entity_set = set()
    relation_set = set()
    entity_context_dict = dict()
    for (h, r, t) in train_set:
        entity_set.add(h)
        entity_set.add(t)
        relation_set.add(r)
        entity_context_dict.setdefault(h, set()).add(t)
        entity_context_dict.setdefault(t, set()).add(h)
    return entity_set, relation_set, entity_context_dict


def construct_text2id_dict(file_name):
    result_dict = dict()
    with open(file_name, 'r', encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            text, id = line.rstrip("\n").split("\t")[:]
            result_dict[text] = int(id)
    return result_dict


def construct_id2text_dict(file_name):
    result_dict = dict()
    with open(file_name, 'r', encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            text, id = line.rstrip("\n").split("\t")[:]
            result_dict[int(id)] = text
    return result_dict


def convert_id_to_text(train_list, dataset):
    text_train_list = list()
    entityid2text_dict = construct_id2text_dict('./data/' + dataset + '/entity2id.txt')
    relationid2text_dict = construct_id2text_dict('./data/' + dataset + '/relation2id.txt')
    for (h, r, t) in train_list:
        text_train_list.append((entityid2text_dict[h], relationid2text_dict[r], entityid2text_dict[t]))
    return text_train_list


def get_1or2_path_from_head(head_ent, rel, entity_adj_table_with_rel):
    paths = {}  # actually second-order + first-order, {entity: [[edge]]}
    first_order_entity = set()
    first_order_relation = dict()

    if head_ent not in entity_adj_table_with_rel:
        return paths
    for tail_entity, relation in entity_adj_table_with_rel[head_ent]:
        first_order_entity.add(tail_entity)
        if relation != rel:
            if tail_entity in paths:
                paths[tail_entity].append([relation])
            else:
                paths[tail_entity] = [[relation]]

        if tail_entity in first_order_relation:
            first_order_relation[tail_entity].append(relation)
        else:
            first_order_relation[tail_entity] = [relation]

    for node in first_order_entity:
        if node not in entity_adj_table_with_rel:
            continue
        for tail_entity, relation in entity_adj_table_with_rel[node]:
            if tail_entity in paths:
                for r in first_order_relation[node]:
                    paths[tail_entity].append([r, relation])
            else:
                paths[tail_entity] = []
                for r in first_order_relation[node]:
                    paths[tail_entity].append([r, relation])

    return paths  # {entity: [[edge]]}


def find_relation_context(h, r, t, entity_adj_table_with_rel):
    tail_ent2paths = get_1or2_path_from_head(h, r, entity_adj_table_with_rel)
    return tail_ent2paths.get(t, [])


def construct_adj_table(train_list, entity_total, relation_total, max_context):
    entity_adj_table_with_rel = dict()  # {head_entity: [(tail_entity, relation)]}
    entity_adj_table = dict()  # {head_entity: [tail_entity]}
    relation_adj_table = dict()  # {relation: [[edge]]}

    for train_data in train_list:
        h, r, t = train_data
        entity_adj_table.setdefault(h, set()).add(t)
        entity_adj_table.setdefault(t, set()).add(h)
        entity_adj_table_with_rel.setdefault(h, list()).append((t, r))

    for train_data in train_list:
        h, r, t = train_data
        paths = find_relation_context(h, r, t, entity_adj_table_with_rel)
        relation_adj_table.setdefault(r, []).extend(paths)
    for k, v in relation_adj_table.items():
        relation_adj_table[k] = set([tuple(i) for i in v])

    max_context_num = max_context
    for k, v in entity_adj_table.items():
        if len(v) > max_context_num:
            res = list(v)
            res = res[:max_context_num]
            entity_adj_table[k] = set(res)
    for k, v in relation_adj_table.items():
        if len(v) > max_context_num:
            res = list(v)
            res = res[:max_context_num]
            relation_adj_table[k] = set(res)

    entity_DAD = torch.Tensor(entity_total, max_context_num + 1, max_context_num + 1).cuda()
    relation_DAD = torch.Tensor(relation_total, max_context_num + 1, max_context_num + 1).cuda()

    for entity in range(entity_total):
        A = torch.eye(max_context_num + 1, max_context_num + 1).cuda()
        tmp = torch.ones(max_context_num + 1).cuda()
        A[0, :max_context_num + 1] = tmp
        A[:max_context_num + 1, 0] = tmp

        D = np.eye(max_context_num + 1, max_context_num + 1)
        i = list(range(max_context_num + 1))
        D[i, i] = 2
        D[0][0] = max_context_num + 1

        if entity in entity_adj_table:
            neighbours_list = list(entity_adj_table[entity])
            for index, neighbour in enumerate(neighbours_list):
                if neighbour not in entity_adj_table:
                    continue
                for index2, neighbour2 in enumerate(neighbours_list):
                    if index == index2:
                        continue
                    if neighbour2 in entity_adj_table[neighbour]:
                        A[index+1, index2+1] = 1
                        D[index+1][index+1] += 1

        D = np.linalg.inv(D)
        D = torch.Tensor(D).cuda()
        D[i, i] = torch.sqrt(D[i, i])

        entity_DAD[entity] = D.mm(A).mm(D)

    for relation in range(relation_total):
        A = torch.eye(max_context_num + 1, max_context_num + 1).cuda()
        tmp = torch.ones(max_context_num + 1).cuda()
        A[0, :max_context_num + 1] = tmp
        A[:max_context_num + 1, 0] = tmp

        D = np.eye(max_context_num + 1, max_context_num + 1)
        i = list(range(max_context_num + 1))
        D[i, i] = 2
        D[0][0] = max_context_num + 1

        if relation in relation_adj_table:
            neighbours_set = relation_adj_table[relation]
            for index, neighbour in enumerate(neighbours_set):
                if len(neighbour) != 1:
                    continue
                if neighbour[0] not in relation_adj_table:
                    continue
                adj_set = relation_adj_table[neighbour[0]]
                for index2, neighbour2 in enumerate(neighbours_set):
                    if index == index2:
                        continue
                    if neighbour2 in adj_set:
                        A[index+1, index2+1] = 1
                        D[index+1][index+1] += 1
        D = np.linalg.inv(D)
        D = torch.Tensor(D).cuda()
        i = list(range(max_context_num + 1))
        D[i, i] = torch.sqrt(D[i, i])

        relation_DAD[relation] = D.mm(A).mm(D)

    for k, v in entity_adj_table.items():
        res = list(v)
        entity_adj_table[k] = res + [entity_total] * (max_context_num - len(res))  # 补padding

    for k, v in relation_adj_table.items():
        res = []
        for i in v:
            if len(i) == 1:
                res.extend(list(i))
                res.append(relation_total)
            else:
                res.extend(list(i))

        relation_adj_table[k] = res + [relation_total] * 2 * (max_context_num - len(res) // 2)  # 补padding

    return entity_adj_table, relation_adj_table, max_context_num, entity_DAD, relation_DAD
    # return entity_adj_table, max_context_num, entity_DAD


def bern_sampling_prepare(train_list):
    head2count = dict()
    tail2count = dict()
    for h, r, t in train_list:
        head2count[h] = head2count.get(h, 0) + 1
        tail2count[t] = tail2count.get(t, 0) + 1

    hpt = 0.0  # head per tail
    for t, count in tail2count.items():
        hpt += count
    hpt /= len(tail2count)

    tph = 0.0
    for h, count in head2count.items():
        tph += count
    tph /= len(head2count)

    return tph, hpt


def one_negative_sampling(golden_triple, train_set, entity_total, bern=True, tph=0.0, hpt=0.0):
    h, r, t = golden_triple
    if not bern:  # uniform sampling
        while True:
            e = random.randint(0, entity_total - 1)
            is_head = random.randint(0, 1)
            if is_head:
                if (e, r, t) in train_set:
                    continue
                else:
                    negative_triple = (e, r, t)
                    break
            else:
                if (h, r, e) in train_set:
                    continue
                else:
                    negative_triple = (h, r, e)
                    break
    else:
        sampling_head_prob = tph / (tph + hpt)
        while True:
            e = random.randint(0, entity_total - 1)
            is_head = random.random() > sampling_head_prob
            if is_head:
                if (e, r, t) in train_set:
                    continue
                else:
                    negative_triple = (e, r, t)
                    break
            else:
                if (h, r, e) in train_set:
                    continue
                else:
                    negative_triple = (h, r, e)
                    break

    return negative_triple


def get_batch(batch_size, batch, epoch, phs, prs, pts, nhs, nrs, nts):
    r = min((batch + 1) * batch_size, len(phs))

    return (phs[batch * batch_size: r], prs[batch * batch_size: r], pts[batch * batch_size: r]), \
           (nhs[epoch, batch * batch_size: r], nrs[epoch, batch * batch_size: r], nts[epoch, batch * batch_size: r])


def get_batch_A(triples, entity_A, relation_A):
    h, r, t = triples
    return entity_A[h.cpu().numpy()], relation_A[r.cpu().numpy()], entity_A[t.cpu().numpy()]