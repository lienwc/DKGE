import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import json
import os
import time

import config
import validate
import test

# gpu_ids = [0, 1]


class DynamicKGE(nn.Module):
    def __init__(self, config):
        super(DynamicKGE, self).__init__()

        self.entity_emb = nn.Parameter(torch.Tensor(config.entity_total, config.dim))
        self.relation_emb = nn.Parameter(torch.Tensor(config.relation_total, config.dim))

        # self.entity_context = nn.Parameter(torch.Tensor(config.entity_total + 1, config.dim), )
        # self.relation_context = nn.Parameter(torch.Tensor(config.relation_total + 1, config.dim))
        self.entity_context = nn.Embedding(config.entity_total + 1, config.dim, padding_idx=config.entity_total)
        self.relation_context = nn.Embedding(config.relation_total + 1, config.dim, padding_idx=config.relation_total)

        self.entity_gcn_weight = nn.Parameter(torch.Tensor(config.dim, config.dim))
        self.relation_gcn_weight = nn.Parameter(torch.Tensor(config.dim, config.dim))

        self.gate_entity = nn.Parameter(torch.Tensor(config.dim))
        self.gate_relation = nn.Parameter(torch.Tensor(config.dim))

        self.v_ent = nn.Parameter(torch.Tensor(config.dim))
        self.v_rel = nn.Parameter(torch.Tensor(config.dim))

        self.pht_o = dict()
        self.pr_o = dict()

        if config.init_with_DKGE:
            self._init_parameters_with_DKGE()
        else:
            self._init_parameters()

    def _init_parameters(self):
        if config.init_with_transe:
            transe_entity_emb, transe_relation_emb = config.get_transe_embdding()
            self.entity_emb.data = transe_entity_emb
            self.relation_emb.data = transe_relation_emb
        else:
            nn.init.xavier_uniform_(self.entity_emb.data)
            nn.init.xavier_uniform_(self.relation_emb.data)

        nn.init.uniform_(self.gate_entity.data)
        nn.init.uniform_(self.gate_relation.data)
        nn.init.uniform_(self.v_ent.data)
        nn.init.uniform_(self.v_rel.data)

        stdv = 1. / math.sqrt(self.entity_gcn_weight.size(1))
        self.entity_gcn_weight.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.relation_gcn_weight.size(1))
        self.relation_gcn_weight.data.uniform_(-stdv, stdv)

    def _init_parameters_with_DKGE(self):
        entity_emb, relation_emb, entity_context, relation_context, \
        entity_gcn_weight, relation_gcn_weight, gate_entity, gate_relation, \
        v_entity, v_relation = config.load_parameters(200)

        entity_mapping_dict, relation_mapping_dict = config.construct_snapshots_mapping_dict()[4:]

        self.entity_gcn_weight.data = entity_gcn_weight
        self.relation_gcn_weight.data = relation_gcn_weight
        self.gate_entity.data = gate_entity
        self.gate_relation.data = gate_relation
        self.v_ent.data = v_entity
        self.v_rel.data = v_relation

        for id1, id2 in entity_mapping_dict.items():
            self.entity_emb.data[id2] = entity_emb[id1]
            self.entity_context.weight.data[id2] = entity_context[id1]

        for id1, id2 in relation_mapping_dict.items():
            self.relation_emb.data[id2] = relation_emb[id1]
            self.relation_context.weight.data[id2] = relation_context[id1]

    def _calc(self, h, t, r):
        return torch.norm(h + r - t, p=config.norm, dim=1)

    def get_entity_context(self, entities):
        '''
        :param entities: [e, ..., e]
        :return:
        '''
        entities_context = []
        for e in entities:
            entities_context.extend(config.entity_adj_table.get(int(e), [config.entity_total] * config.max_context_num))
        # return entities_context
        return torch.LongTensor(entities_context).cuda()

    def get_relation_context(self, relations):
        relations_context = []
        for r in relations:
            relations_context.extend(
                config.relation_adj_table.get(int(r), [config.relation_total] * 2 * config.max_context_num))
        # return relations_context
        return torch.LongTensor(relations_context).cuda()

    def get_adj_entity_vec(self, entity_vec_list, adj_entity_list):
        # adj_entity_vec_list = self.entity_context[adj_entity_list]
        adj_entity_vec_list = self.entity_context(adj_entity_list)
        adj_entity_vec_list = adj_entity_vec_list.view(-1, config.max_context_num, config.dim)

        return torch.cat((entity_vec_list.unsqueeze(1), adj_entity_vec_list), dim=1)

    def get_adj_relation_vec(self, relation_vec_list, adj_relation_list):
        # adj_relation_vec_list = self.relation_context[adj_relation_list]
        adj_relation_vec_list = self.relation_context(adj_relation_list)
        adj_relation_vec_list = adj_relation_vec_list.view(-1, config.max_context_num, 2,
                                                           config.dim).cuda()
        adj_relation_vec_list = torch.sum(adj_relation_vec_list, dim=2)

        return torch.cat((relation_vec_list.unsqueeze(1), adj_relation_vec_list), dim=1)

    def score(self, o, adj_vec_list, target='entity'):
        os = torch.cat(tuple([o] * (config.max_context_num+1)), dim=1).reshape(-1, config.max_context_num+1, config.dim)
        tmp = F.relu(torch.mul(adj_vec_list, os), inplace=False)  # batch x max x 2dim
        if target == 'entity':
            score = torch.matmul(tmp, self.v_ent)  # batch x max
        else:
            score = torch.matmul(tmp, self.v_rel)
        return score

    def calc_subgraph_vec(self, o, adj_vec_list, target="entity"):
        alpha = self.score(o, adj_vec_list, target)
        alpha = F.softmax(alpha)

        sg = torch.sum(torch.mul(torch.unsqueeze(alpha, dim=2), adj_vec_list), dim=1)  # batch x dim
        return sg

    def gcn(self, A, H, target='entity'):
        support = torch.matmul(A, H)
        if target == 'entity':
            output = F.relu(torch.matmul(support, self.entity_gcn_weight))
        elif target == 'relation':
            output = F.relu(torch.matmul(support, self.relation_gcn_weight))
        return output

    def save_parameters(self, file_name, epoch):
        if not os.path.exists(config.res_dir):
            os.mkdir(config.res_dir)

        ent_f = open(config.res_dir + 'entity_o_' + file_name + str(epoch), "w")
        ent_f.write(json.dumps(self.pht_o))
        ent_f.close()

        rel_f = open(config.res_dir + 'relation_o_' + file_name + str(epoch), "w")
        rel_f.write(json.dumps(self.pr_o))
        rel_f.close()

        para2vec = {}
        lists = self.state_dict()
        for var_name in lists:
            para2vec[var_name] = lists[var_name].cpu().numpy().tolist()

        f = open(config.res_dir + 'all_' + file_name + str(epoch), "w")
        f.write(json.dumps(para2vec))
        f.close()

    def save_phrt_o(self, pos_h, pos_r, pos_t, ph_o, pr_o, pt_o):
        for i in range(len(pos_h)):
            h = str(int(pos_h[i]))
            self.pht_o[h] = ph_o[i].detach().cpu().numpy().tolist()

            t = str(int(pos_t[i]))
            self.pht_o[t] = pt_o[i].detach().cpu().numpy().tolist()

            r = str(int(pos_r[i]))
            self.pr_o[r] = pr_o[i].detach().cpu().numpy().tolist()

    def forward(self, epoch, golden_triples, negative_triples, ph_A, pr_A, pt_A, nh_A, nr_A, nt_A):
        # multi golden and multi negative
        pos_h, pos_r, pos_t = golden_triples
        neg_h, neg_r, neg_t = negative_triples

        p_h = self.entity_emb[pos_h.cpu().numpy()]
        p_t = self.entity_emb[pos_t.cpu().numpy()]
        p_r = self.relation_emb[pos_r.cpu().numpy()]
        n_h = self.entity_emb[neg_h.cpu().numpy()]
        n_t = self.entity_emb[neg_t.cpu().numpy()]
        n_r = self.relation_emb[neg_r.cpu().numpy()]

        ph_adj_entity_list = self.get_entity_context(pos_h)
        pt_adj_entity_list = self.get_entity_context(pos_t)
        nh_adj_entity_list = self.get_entity_context(neg_h)
        nt_adj_entity_list = self.get_entity_context(neg_t)
        pr_adj_relation_list = self.get_relation_context(pos_r)
        nr_adj_relation_list = self.get_relation_context(neg_r)
        
        ph_adj_entity_vec_list = self.get_adj_entity_vec(p_h, ph_adj_entity_list)
        pt_adj_entity_vec_list = self.get_adj_entity_vec(p_t, pt_adj_entity_list)
        nh_adj_entity_vec_list = self.get_adj_entity_vec(n_h, nh_adj_entity_list)
        nt_adj_entity_vec_list = self.get_adj_entity_vec(n_t, nt_adj_entity_list)
        pr_adj_relation_vec_list = self.get_adj_relation_vec(p_r, pr_adj_relation_list)
        nr_adj_relation_vec_list = self.get_adj_relation_vec(n_r, nr_adj_relation_list)

        # gcn
        ph_adj_entity_vec_list = self.gcn(ph_A, ph_adj_entity_vec_list, target='entity')
        pt_adj_entity_vec_list = self.gcn(pt_A, pt_adj_entity_vec_list, target='entity')
        nh_adj_entity_vec_list = self.gcn(nh_A, nh_adj_entity_vec_list, target='entity')
        nt_adj_entity_vec_list = self.gcn(nt_A, nt_adj_entity_vec_list, target='entity')
        pr_adj_relation_vec_list = self.gcn(pr_A, pr_adj_relation_vec_list, target='relation')
        nr_adj_relation_vec_list = self.gcn(nr_A, nr_adj_relation_vec_list, target='relation')

        ph_sg = self.calc_subgraph_vec(p_h, ph_adj_entity_vec_list, target='entity')
        pt_sg = self.calc_subgraph_vec(p_t, pt_adj_entity_vec_list, target='entity')
        nh_sg = self.calc_subgraph_vec(n_h, nh_adj_entity_vec_list, target='entity')
        nt_sg = self.calc_subgraph_vec(n_t, nt_adj_entity_vec_list, target='entity')
        pr_sg = self.calc_subgraph_vec(p_r, pr_adj_relation_vec_list, target='relation')
        nr_sg = self.calc_subgraph_vec(n_r, nr_adj_relation_vec_list, target='relation')

        ph_o = torch.mul(F.sigmoid(self.gate_entity), p_h) + torch.mul(1 - F.sigmoid(self.gate_entity), ph_sg)
        pt_o = torch.mul(F.sigmoid(self.gate_entity), p_t) + torch.mul(1 - F.sigmoid(self.gate_entity), pt_sg)
        nh_o = torch.mul(F.sigmoid(self.gate_entity), n_h) + torch.mul(1 - F.sigmoid(self.gate_entity), nh_sg)
        nt_o = torch.mul(F.sigmoid(self.gate_entity), n_t) + torch.mul(1 - F.sigmoid(self.gate_entity), nt_sg)
        pr_o = torch.mul(F.sigmoid(self.gate_relation), p_r) + torch.mul(1 - F.sigmoid(self.gate_relation), pr_sg)
        nr_o = torch.mul(F.sigmoid(self.gate_relation), n_r) + torch.mul(1 - F.sigmoid(self.gate_relation), nr_sg)
        # ph_o = p_h
        # nh_o = n_h
        # pt_o = p_t
        # nt_o = n_t
        # pr_o = p_r
        # nr_o = n_r

        # score for loss
        p_score = self._calc(ph_o, pt_o, pr_o)
        n_score = self._calc(nh_o, nt_o, nr_o)

        if epoch % config.validation_step == 0 and epoch != 0:
            self.save_phrt_o(pos_h, pos_r, pos_t, ph_o, pr_o, pt_o)
        else:
            self.pht_o.clear()
            self.pr_o.clear()

        return p_score, n_score


def main():
    print('preparing data...')
    phs, prs, pts, nhs, nrs, nts = config.prepare_data()
    print('preparing data complete')

    print('train starting...')

    dynamicKGE = DynamicKGE(config).cuda()

    # DataParallel
    # dynamicKGE.to(gpu_ids[0])
    # dynamicKGE = torch.nn.DataParallel(dynamicKGE, device_ids=gpu_ids)

    # optimizer = optim.SGD(dynamicKGE.parameters(), lr=config.learning_rate)
    optimizer = optim.Adam(dynamicKGE.parameters(), lr=config.learning_rate)
    # optimizer = optim.Adagrad(dynamicKGE.parameters(), lr=config.learning_rate)
    criterion = nn.MarginRankingLoss(config.margin, False).cuda()

    best_filter_mrr = 0.0
    best_epoch = 0
    bad_count = 0
    bad_patience = 3
    for epoch in range(config.train_times):
        # print(dynamicKGE.entity_context(torch.LongTensor([0]).cuda()))
        # print(dynamicKGE.entity_context(torch.LongTensor([config.entity_total-1]).cuda()))
        # print(dynamicKGE.entity_context(torch.LongTensor([config.entity_total]).cuda()))
        print('----------training the ' + str(epoch) + ' epoch----------')
        start_time = time.time()
        epoch_avg_loss = 0.0
        for batch in range(config.nbatchs):
            optimizer.zero_grad()
            golden_triples, negative_triples = config.get_batch(batch, epoch, phs, prs, pts, nhs, nrs, nts)
            ph_A, pr_A, pt_A = config.get_batch_A(golden_triples)
            nh_A, nr_A, nt_A = config.get_batch_A(negative_triples)

            p_scores, n_scores = dynamicKGE(epoch, golden_triples, negative_triples, ph_A, pr_A, pt_A, nh_A, nr_A, nt_A)
            y = torch.Tensor([-1]).cuda()
            loss = criterion(p_scores, n_scores, y)

            loss.backward()
            optimizer.step()

            epoch_avg_loss += (float(loss.item()) / config.nbatchs)
            torch.cuda.empty_cache()

        print('----------epoch avg loss: ' + str(epoch_avg_loss) + ' ----------')
        end_time = time.time()
        print('----------epoch time: ' + str(end_time - start_time) + ' ----------')

        if epoch % config.validation_step == 0 and epoch != 0:
            # dynamicKGE.module.save_parameters('parameters', epoch)
            dynamicKGE.save_parameters('parameters', epoch)

            print("Validating...")
            filter_mrr = validate.validate(dynamicKGE.pht_o, dynamicKGE.pr_o)
            if filter_mrr > best_filter_mrr:
                best_filter_mrr = filter_mrr
                best_epoch = epoch
                bad_count = 0
            else:
                bad_count += 1
            print("Best MRR:%.3f; Current MRR:%.3f; Bad count:%d" % (best_filter_mrr, filter_mrr, bad_count))

            if bad_count == bad_patience:
                print("Early stopped at epoch %s" % str(epoch))
                print("The best epoch is: %s" % str(best_epoch))
                print("The best MRR is: %s" % str(best_filter_mrr))
                break

    print('train ending...')

    entity_emb, relation_emb = config.load_o_emb(best_epoch, input=False)
    print('test link prediction starting...')
    test.test_link_prediction(config.test_list, set(config.train_list), entity_emb, relation_emb)
    print('test link prediction ending...')


main()
