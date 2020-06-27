from util.train_util import *


def get_affected_entities(entity_set1, entity_set2, entity_context_dict1, entity_context_dict2):
    affected_entities_set1 = set()
    for entity in entity_set2 - entity_set1:
        for target_entity, context_set in entity_context_dict2.items():
            if entity in context_set:
                affected_entities_set1.add(target_entity)

    affected_entities_set1 = affected_entities_set1 & entity_set1

    affected_entities_set2 = set()
    for entity in entity_set1 - entity_set2:
        for target_entity, context_set in entity_context_dict1.items():
            if entity in context_set:
                affected_entities_set2.add(target_entity)
    affected_entities_set2 = affected_entities_set2 & entity_set2

    return affected_entities_set1 | affected_entities_set2


def get_affected_relations(snapshot2_train_set, relation_set1, relation_set2):
    relation_dict = dict()
    head_entity_dict = dict()
    relation_context_dict = dict()

    for (h, r, t) in snapshot2_train_set:
        head_entity_dict.setdefault(h, list()).append([r, t])
        relation_dict.setdefault(r, list()).append([h, t])

    for relation, nodes_set in relation_dict.items():
        relation_context_dict[relation] = set()
        for nodes in nodes_set:
            head, tail = nodes[:2]
            # 一阶关系
            for edge in head_entity_dict[head]:
                if edge[1] == tail:
                    relation_context_dict[relation].add(edge[0])
            relation_context_dict[relation].remove(relation)

            # 二阶关系
            for edge1 in head_entity_dict[head]:
                if edge1[1] in head_entity_dict:
                    for edge2 in head_entity_dict[edge1[1]]:
                        if edge2[1] == tail:
                            relation_context_dict[relation].add("%s+%s" % (edge1[0], edge2[0]))

    affected_relations_set = set()
    for new_relation in (relation_set2 - relation_set1):
        for relation, contexts_set in relation_context_dict.items():
            for context_relation in contexts_set:
                if new_relation in context_relation:
                    affected_relations_set.add(relation)
                    break

    affected_relations_set = affected_relations_set - (relation_set2 - relation_set1)
    return affected_relations_set


def analyse_affected_data(dataset_v1, dataset_v2):
    snapshot1_train_list = read_file(file_name='./data/' + dataset_v1 + '/train2id.txt')
    snapshot2_train_list = read_file(file_name='./data/' + dataset_v2 + '/train2id.txt')

    snapshot1_train_set = set(convert_id_to_text(snapshot1_train_list, dataset_v1))
    snapshot2_train_set = set(convert_id_to_text(snapshot2_train_list, dataset_v2))

    entity_set1, relation_set1, entity_context_dict1 = get_basic_info(snapshot1_train_set)
    entity_set2, relation_set2, entity_context_dict2 = get_basic_info(snapshot2_train_set)

    affected_entities = get_affected_entities(entity_set1, entity_set2, entity_context_dict1, entity_context_dict2)
    affected_relations = get_affected_relations(snapshot2_train_set, relation_set1, relation_set2)

    new_entities = entity_set2 - entity_set1
    new_relations = relation_set2 - relation_set1

    affected_triples = list()
    for (h, r, t) in snapshot2_train_set:
        if h in affected_entities or h in new_entities or t in affected_entities or t in new_entities or \
                r in affected_relations or r in new_relations:
            affected_triples.append((h, r, t))

    return affected_entities, affected_relations, affected_triples


def construct_snapshots_mapping_dict(dataset_v1, dataset_v2):
    snapshot1_entity2id_dict = construct_text2id_dict(file_name='./data/' + dataset_v1 + '/entity2id.txt')
    snapshot1_relation2id_dict = construct_text2id_dict(file_name='./data/' + dataset_v1 + '/relation2id.txt')

    snapshot2_entity2id_dict = construct_text2id_dict(file_name='./data/' + dataset_v2 + '/entity2id.txt')
    snapshot2_relation2id_dict = construct_text2id_dict(file_name='./data/' + dataset_v2 + '/relation2id.txt')

    entity_mapping_dict = dict()
    for entity, id in snapshot1_entity2id_dict.items():
        if entity in snapshot2_entity2id_dict:
            entity_mapping_dict[id] = snapshot2_entity2id_dict[entity]

    relation_mapping_dict = dict()
    for relation, id in snapshot1_relation2id_dict.items():
        if relation in snapshot2_relation2id_dict:
            relation_mapping_dict[id] = snapshot2_relation2id_dict[relation]

    return snapshot1_entity2id_dict, snapshot1_relation2id_dict, snapshot2_entity2id_dict, snapshot2_relation2id_dict, \
        entity_mapping_dict, relation_mapping_dict


def analyse_snapshots(dataset_v1, dataset_v2):
    snapshot1_entity2id_dict, snapshot1_relation2id_dict, snapshot2_entity2id_dict, snapshot2_relation2id_dict, \
     entity_mapping_dict, relation_mapping_dict = construct_snapshots_mapping_dict(dataset_v1, dataset_v2)

    added_entities = set()
    for entity, id in snapshot2_entity2id_dict.items():
        if entity not in snapshot1_entity2id_dict:
            added_entities.add(id)

    added_relations = set()
    for relation, id in snapshot2_relation2id_dict.items():
        if relation not in snapshot1_relation2id_dict:
            added_relations.add(id)

    affected_entities, affected_relations, affected_triples = analyse_affected_data(dataset_v1, dataset_v2)

    affected_entities = set([snapshot2_entity2id_dict[e] for e in affected_entities])
    affected_relations = set([snapshot2_relation2id_dict[r] for r in affected_relations])
    affected_triples = [(snapshot2_entity2id_dict[h], snapshot2_relation2id_dict[r], snapshot2_entity2id_dict[t]) for (h, r, t) in affected_triples]

    print("affected entities:%d" % len(affected_entities))
    print("affected relations:%d" % len(affected_relations))
    print("affected triples:%d" % len(affected_triples))
    print("added entities:%d" % len(added_entities))
    print("addded relations:%d" % len(added_relations))

    return affected_entities, affected_relations, affected_triples, added_entities, added_relations, entity_mapping_dict, relation_mapping_dict
