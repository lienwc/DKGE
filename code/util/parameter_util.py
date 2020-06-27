import os
import json
import torch


def load_o_emb(res_path, entity_total, relation_total, dim, input=False):
    entity_o_path = os.path.join(res_path, 'entity_o')
    relation_o_path = os.path.join(res_path, 'relation_o')
    if input:
        with open(entity_o_path, "r") as f:
            emb = json.loads(f.read())
            # entity_emb = torch.Tensor(len(emb), dim)
            entity_emb = torch.Tensor(entity_total, dim)
            for k, v in emb.items():
                entity_emb[int(k)] = torch.Tensor(v)

        with open(relation_o_path, "r") as f:
            emb = json.loads(f.read())
            # relation_emb = torch.Tensor(len(emb), dim)
            relation_emb = torch.Tensor(relation_total, dim)
            for k, v in emb.items():
                relation_emb[int(k)] = torch.Tensor(v)
        return entity_emb.cuda(), relation_emb.cuda()

    else:
        with open(entity_o_path, "r") as f:
            ent_emb = json.loads(f.read())
        with open(relation_o_path, "r") as f:
            rel_emb = json.loads(f.read())

        # entity_emb = torch.Tensor(len(ent_emb), dim)
        entity_emb = torch.Tensor(entity_total, dim)
        for k, v in ent_emb.items():
            entity_emb[int(k)] = torch.Tensor(v)

        # relation_emb = torch.Tensor(len(rel_emb), dim)
        relation_emb = torch.Tensor(relation_total, dim)
        for k, v in rel_emb.items():
            relation_emb[int(k)] = torch.Tensor(v)

        return entity_emb.cuda(), relation_emb.cuda()


def load_parameters(parameter_path):
    with open(os.path.join(parameter_path, "all_parameters"), 'r') as f:
        emb = json.loads(f.read())
        entity_emb = emb.get("entity_emb", emb['entity_emb.weight'])
        relation_emb = emb.get('relation_emb', emb['relation_emb.weight'])
        entity_context = emb['entity_context.weight']
        relation_context = emb['relation_context.weight']
        entity_gcn_weight = emb['entity_gcn_weight']
        relation_gcn_weight = emb['relation_gcn_weight']
        gate_entity = emb['gate_entity']
        gate_relation = emb['gate_relation']
        v_ent = emb['v_ent']
        v_rel = emb['v_rel']

        # give index, return embedding
        return torch.Tensor(entity_emb).cuda(), torch.Tensor(relation_emb).cuda(), \
               torch.Tensor(entity_context).cuda(), torch.Tensor(relation_context).cuda(), \
               torch.Tensor(entity_gcn_weight).cuda(), torch.Tensor(relation_gcn_weight).cuda(), \
               torch.Tensor(gate_entity).cuda(), torch.Tensor(gate_relation).cuda(), \
               torch.Tensor(v_ent).cuda(), torch.Tensor(v_rel).cuda()
