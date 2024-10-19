import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FGKAN(nn.Module):
    def __init__(self, args, n_entity, n_relation):
        super(FGKAN, self).__init__()
        self._parse_args(args, n_entity, n_relation)
        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)

        # 注意力层
        self.attention_layer = nn.Linear(2 * self.dim, 1)

        # 超参数
        self.ssl_temp = 0.05
        self.kge_weight = 1e-8
        self._init_weight()
        self.n_layer = 2

    def forward(
        self,
        items: torch.LongTensor,
        kg_init_triple_set: list,
        # 省略的参数
    ):
        kg_embeddings = []
        
        # 初始嵌入
        kg_emb_0 = self.entity_emb(kg_init_triple_set[0][0])
        kg_initial_embedding = kg_emb_0.mean(dim=1)
        kg_embeddings.append(kg_initial_embedding)

        # 多层图卷积
        for i in range(self.n_layer):
            # 当前层的实体嵌入
            h_set_emb = self.entity_emb(kg_init_triple_set[0][0])
            path_emb = self.relation_emb(kg_init_triple_set[1][0])

            for k in range(1, i + 1):
                # 使用图卷积计算新的节点嵌入
                h_set_emb = self.graph_convolution(h_set_emb, path_emb, kg_init_triple_set[0][k])

            # 目标实体嵌入
            t_emb = self.entity_emb(kg_init_triple_set[2][i])

            # 使用注意力机制加权融合
            kg_emb_i = self._knowledge_attention(h_set_emb, path_emb, t_emb)
            kg_embeddings.append(kg_emb_i)

        return kg_embeddings

    def graph_convolution(self, h_set_emb, path_emb, neighbors):
        # 计算邻居的嵌入
        neighbor_emb = self.entity_emb(neighbors)  # 获取邻居的嵌入
        combined_emb = torch.cat((h_set_emb.unsqueeze(1).expand(-1, neighbor_emb.size(1), -1), neighbor_emb), dim=-1)  # 拼接当前节点和邻居节点嵌入

        # 注意力计算
        attention_weights = F.softmax(self.attention_layer(combined_emb), dim=1)  # 计算注意力权重
        h_set_emb = torch.bmm(attention_weights.transpose(1, 2), neighbor_emb)  # 应用注意力权重

        return h_set_emb

    def _knowledge_attention(self, h_set_emb, path_emb, t_emb):
        # 这里可以继续实现注意力机制的具体细节
        # 例如，对 h_set_emb 和 t_emb 进行加权
        return h_set_emb  # 返回加权后的嵌入

"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class FGKAN(nn.Module):
    def __init__(self, args, n_entity, n_relation):
        super(FGKAN, self).__init__()
        self._parse_args(args, n_entity, n_relation)
        self.entity_emb = nn.Embedding(self.n_entity, self.dim)  # 将实体和关系嵌入指定维度的嵌入向量
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        self.attention = nn.Sequential(
                nn.Linear(2*self.dim, self.dim, bias=False),
                nn.Sigmoid(),
                nn.Linear(self.dim, 1, bias=False),
                nn.Sigmoid(),
                )
        # parameters for kge enbedding with graph contrastive learning
        self.ssl_temp = 0.05    # for softmax
        self.kge_weight = 1e-8  # for kge_loss  初始化用于图对比学习的超参数，包括softmax的温度参数和KGE损失的权重参数。
        self._init_weight()
        self.n_layer = 2

# 实现了模型的前向传播过程
    def forward(
        self,
        items: torch.LongTensor,
        kg_init_triple_set: list,
        ddi_potential_triple_set: list,
        kg_potential_triple_set: list,
        ddi_origin_triple_set: list,
        kg_init_triple_set1: list,
        ddi_potential_triple_set1: list,
        kg_potential_triple_set1: list,
        ddi_origin_triple_set1: list,
        embeddings_0: list,
        embeddings_1: list,
    ):
# 包含(知识图）每一层的嵌入
        kg_embeddings = []
        # [batch_size, triple_set_size, dim]
        kg_emb_0 = self.entity_emb(kg_init_triple_set[0][0])
        # [batch_size, dim]
        kg_intial_embedding = kg_emb_0.mean(dim=1)
        kg_embeddings.append(kg_intial_embedding)
#首先，从kg的初始三元组集合（kg_init_triple_set）中获取初始的实体嵌入（kg_emb_0），然后计算这些实体嵌入的平均值（kg_intial_embedding）并将其添加到知识图嵌入列表中（kg_embeddings）。
        for i in range(self.n_layer):  # 循环遍历当前实体与其他实体之间的关系
            # [batch_size, triple_set_size, dim]
            h_set_emb = self.entity_emb(kg_init_triple_set[0][0])
            # [batch_size, triple_set_size, dim]
            # r_emb = self.relation_emb(kg_triple_set[1][i])
            # 从用户的初始三元组集合中获取当前层的实体嵌入（h_set_emb），表示当前实体与其他实体之间的关系。
            path_emb = self.relation_emb(kg_init_triple_set[1][0])
            # 获取当前层的路径嵌入（path_emb）。
            # [batch_size, triple_set_size, dim]
            for k in range(1, i+1):
                #path_emb += path_emb,self.relation_emb(kg_init_triple_set[1][i])
                h_set_emb += self.entity_emb(kg_init_triple_set[0][i])
                path_emb=torch.mul(path_emb,self.relation_emb(kg_init_triple_set[1][i]))
                # 对于当前层的每个关系，将路径嵌入与关系嵌入相乘（torch.mul）并加到路径嵌入中（path_emb）。
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(kg_init_triple_set[2][i])
            # 获取当前层的目标实体嵌入（t_emb）。
            # [batch_size, triple_set_size,neighbor_size, dim]

            # [batch_size, dim]
            kg_emb_i = self._knowledge_attention(h_set_emb, path_emb, t_emb) # 通过调用 _knowledge_attention 方法对 h_set_emb 和 path_emb 进行加权融合和注意力计算

            kg_embeddings.append(kg_emb_i)

# 最终包含潜在ddi（药物药物图）的每一层的嵌入
        ddi_embeddings = []
        # [batch size, dim]
        ddi_emb_origin = self.entity_emb(items)
        # item_embeddings.append(ddi_emb_origin)
        item_emb_0 = self.entity_emb(ddi_potential_triple_set[0][0])
        ddi_intial_embedding = ddi_emb_origin
        ddi_embeddings.append(ddi_intial_embedding)
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_set_emb = self.entity_emb(ddi_potential_triple_set[0][0])
            # [batch_size, triple_set_size, dim]
            #r_emb = self.relation_emb(ddi_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            path_emb = self.relation_emb(ddi_potential_triple_set[1][0])
            # [batch_size, triple_set_size, dim]
            for k in range(1, i+1):
                h_set_emb += self.entity_emb(ddi_potential_triple_set[0][i])
                #path_emb += self.relation_emb(item_potential_triple_set[1][i])
                path_emb=torch.mul(path_emb,self.relation_emb(ddi_potential_triple_set[1][i]))
            t_emb = self.entity_emb(ddi_potential_triple_set[2][i])
            # [batch_size, triple_set_size,neighbor_size, dim]

            # [batch_size, dim]
            ddi_emb_i = self._knowledge_attention(h_set_emb, path_emb, t_emb)
            ddi_embeddings.append(ddi_emb_i)

# 获得kg（第一个药物）在每一层的嵌入向量
        kg_potential_embeddings = []
        kg_potential_embeddings_0 = self.entity_emb(kg_potential_triple_set[0][0])
        kg_intial_potential_embedding = kg_potential_embeddings_0.mean(dim=1)
        kg_potential_embeddings.append(kg_intial_potential_embedding)
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_set_emb = self.entity_emb(kg_potential_triple_set[0][0])
            # [batch_size, triple_set_size, dim]
            path_emb = self.relation_emb(kg_potential_triple_set[1][0])
            # [batch_size, triple_set_size, dim]
            for k in range(1, i+1):
                h_set_emb += self.entity_emb(kg_potential_triple_set[0][i])
                #path_emb += path_emb,self.relation_emb(kg_potential_triple_set[1][i])
                path_emb=torch.mul(path_emb,self.relation_emb(kg_potential_triple_set[1][i]))
            #r_emb = self.relation_emb(user_potential_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(kg_potential_triple_set[2][i])
            # [batch_size, triple_set_size,neighbor_size, dim]

            # [batch_size, dim]
            kg_potential_emb_i = self._knowledge_attention(h_set_emb, path_emb, t_emb)
            kg_potential_embeddings.append(kg_potential_emb_i)

# ddi原始的项目嵌入
        ddi_origin_embeddings = []
        ddi_origin_embeddings_0 = self.entity_emb(ddi_origin_triple_set[0][0])
        ddi_intial_origin_embedding = ddi_origin_embeddings_0.mean(dim=1)
        ddi_origin_embeddings.append(ddi_intial_origin_embedding)
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_set_emb = self.entity_emb(ddi_origin_triple_set[0][0])
            # [batch_size, triple_set_size, dim]
            path_emb = self.relation_emb(ddi_origin_triple_set[1][0])
            # [batch_size, triple_set_size, dim]
            for k in range(1, i+1):
                h_set_emb += self.entity_emb(ddi_origin_triple_set[0][i])
                #path_emb+= self.relation_emb(ddi_origin_triple_set[1][i])
                path_emb=torch.mul(path_emb,self.relation_emb(ddi_origin_triple_set[1][i]))
            #r_emb = self.relation_emb(ddi_origin_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(ddi_origin_triple_set[2][i])
            # [batch_size, triple_set_size,neighbor_size, dim]

            # [batch_size, dim]
            ddi_origin_emb_i = self._knowledge_attention(h_set_emb, path_emb, t_emb)
            ddi_origin_embeddings.append(ddi_origin_emb_i)





        ##################################药物er
        kg_embeddings1 = []
        # [batch_size, triple_set_size, dim]
        kg_emb_1 = self.entity_emb(kg_init_triple_set1[0][0])
        # [batch_size, dim]
        kg_intial_embedding1 = kg_emb_1.mean(dim=1)
        kg_embeddings1.append(kg_intial_embedding1)
        # 首先，从用户的初始三元组集合（kg_init_triple_set）中获取初始的实体嵌入（kg_emb_0），然后计算这些实体嵌入的平均值（kg_intial_embedding）并将其添加到用户嵌入列表中（kg_embeddings）。
        for i in range(self.n_layer):  # 循环遍历当前实体与其他实体之间的关系
            # [batch_size, triple_set_size, dim]
            h_set_emb = self.entity_emb(kg_init_triple_set1[0][0])
            # [batch_size, triple_set_size, dim]
            # r_emb = self.relation_emb(kg_triple_set[1][i])
            # 从用户的初始三元组集合中获取当前层的实体嵌入（h_set_emb），表示当前实体与其他实体之间的关系。
            path_emb = self.relation_emb(kg_init_triple_set1[1][0])
            # 获取当前层的路径嵌入（path_emb）。
            # [batch_size, triple_set_size, dim]
            for k in range(1, i + 1):
                 # path_emb += path_emb,self.relation_emb(user_init_triple_set[1][i])
                h_set_emb += self.entity_emb(kg_init_triple_set1[0][i])
                path_emb = torch.mul(path_emb, self.relation_emb(kg_init_triple_set1[1][i]))
                # 对于当前层的每个关系，将路径嵌入与关系嵌入相乘（torch.mul）并加到路径嵌入中（path_emb）。
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(kg_init_triple_set1[2][i])
            # 获取当前层的目标实体嵌入（t_emb）。
            # [batch_size, triple_set_size,neighbor_size, dim]
            # [batch_size, dim]
            kg_emb_i1 = self._knowledge_attention(h_set_emb, path_emb,t_emb)  # 通过调用 _knowledge_attention 方法对 h_set_emb 和 path_emb 进行加权融合和注意力计算
            # user_emb_i = self._knowledge_attention(h_set_emb, path_emb, t_emb)
            kg_embeddings1.append(kg_emb_i1)

            # 最终包含潜在项目（药物药物图）的每一层的嵌入
        ddi_embeddings1 = []
        ddi_origin_embeddings = self.entity_emb(ddi_potential_triple_set1[0][0])
        ddi_intial_origin_embedding = ddi_origin_embeddings.mean(dim=1)
        ddi_embeddings1.append(ddi_intial_origin_embedding)
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_set_emb = self.entity_emb(ddi_potential_triple_set1[0][0])
            # [batch_size, triple_set_size, dim]
            # r_emb = self.relation_emb(ddi_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            path_emb = self.relation_emb(ddi_potential_triple_set1[1][0])
            # [batch_size, triple_set_size, dim]
            for k in range(1, i + 1):
                h_set_emb += self.entity_emb(ddi_potential_triple_set1[0][i])
                # path_emb += self.relation_emb(ddi_potential_triple_set[1][i])
                path_emb = torch.mul(path_emb, self.relation_emb(ddi_potential_triple_set1[1][i]))
            t_emb = self.entity_emb(ddi_potential_triple_set1[2][i])
            # [batch_size, triple_set_size,neighbor_size, dim]
            # [batch_size, dim]
            ddi_emb_i1 = self._knowledge_attention(h_set_emb, path_emb, t_emb)
            ddi_embeddings1.append(ddi_emb_i1)

            # 获得知识图（潜在）在每一层的嵌入向量
        kg_potential_embeddings1 = []
        kg_potential_embeddings_0 = self.entity_emb(kg_potential_triple_set1[0][0])
        kg_intial_potential_embedding = kg_potential_embeddings_0.mean(dim=1)
        kg_potential_embeddings1.append(kg_intial_potential_embedding)
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_set_emb = self.entity_emb(kg_potential_triple_set1[0][0])
            # [batch_size, triple_set_size, dim]
            path_emb = self.relation_emb(kg_potential_triple_set1[1][0])
            # [batch_size, triple_set_size, dim]
            for k in range(1, i + 1):
                h_set_emb += self.entity_emb(kg_potential_triple_set1[0][i])
                # path_emb += path_emb,self.relation_emb(kg_potential_triple_set[1][i])
                path_emb = torch.mul(path_emb, self.relation_emb(kg_potential_triple_set1[1][i]))
            # r_emb = self.relation_emb(kg_potential_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(kg_potential_triple_set1[2][i])
            # [batch_size, triple_set_size,neighbor_size, dim]
            # [batch_size, dim]
            kg_potential_emb_i1 = self._knowledge_attention(h_set_emb, path_emb, t_emb)
            kg_potential_embeddings1.append(kg_potential_emb_i1)

            # ddi原始的项目嵌入
        ddi_origin_embeddings1 = []
        ddi_origin_embeddings_0 = self.entity_emb(ddi_origin_triple_set1[0][0])
        ddi_intial_origin_embedding = ddi_origin_embeddings_0.mean(dim=1)
        ddi_origin_embeddings1.append(ddi_intial_origin_embedding)
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_set_emb = self.entity_emb(ddi_origin_triple_set1[0][0])
            # [batch_size, triple_set_size, dim]
            path_emb = self.relation_emb(ddi_origin_triple_set1[1][0])
            # [batch_size, triple_set_size, dim]
            for k in range(1, i + 1):
                h_set_emb += self.entity_emb(ddi_origin_triple_set1[0][i])
                # path_emb+= self.relation_emb(ddi_origin_triple_set[1][i])
                path_emb = torch.mul(path_emb, self.relation_emb(ddi_origin_triple_set1[1][i]))
            # r_emb = self.relation_emb(ddi_origin_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(ddi_origin_triple_set1[2][i])
            # [batch_size, triple_set_size,neighbor_size, dim]
            # [batch_size, dim]
            ddi_origin_emb_i1 = self._knowledge_attention(h_set_emb, path_emb, t_emb)
            ddi_origin_embeddings1.append(ddi_origin_emb_i1)
        ddi_origin_embeddings = ddi_embeddings
        if self.n_layer > 0 and (self.agg == 'sum' or self.agg == 'pool'):
             # [batch_size, triple_set_size, dim]
            item_emb_0 = self.entity_emb(ddi_potential_triple_set[0][0])
            # [batch_size, dim]
            ddi_embeddings.append(item_emb_0.mean(dim=1))

        scores = self.predict(kg_embeddings, ddi_embeddings, kg_potential_embeddings, ddi_origin_embeddings,kg_embeddings1, ddi_embeddings1, kg_potential_embeddings1, ddi_origin_embeddings1,embeddings_0,embeddings_1)
        #知识图嵌入，药物嵌入
        return scores



    def predict(self, drug_kg, drug_ddi, drug_kg_two, ddi_origin_embeddings, drug2_kg, drug2_kg_two, drug2_ddi,drug2_ddi_two, embedding_0, embedding_1):
        for i in range(len(drug_kg)):
            drug_kg[i] = torch.cat((drug_kg[i], drug_ddi[i]), dim=-1)
            drug_kg_two[i] = torch.cat((drug_kg_two[i], ddi_origin_embeddings[i]), dim=-1)
        for i in range(len(drug_kg)):
            drug_ddi[i] = torch.cat((drug2_kg[i], drug2_ddi[i]), dim=-1)
            ddi_origin_embeddings[i] = torch.cat((drug2_kg_two[i], drug2_ddi_two[i]), dim=-1)

        e_u = drug_kg[0]
        e_i = drug_ddi[0]
        e_p_u = drug_kg_two[0]
        e_p_i = ddi_origin_embeddings[0]

        kge_loss = 0
        kge_loss1 = 0

        if self.agg == 'concat':
            if len(drug_kg) != len(drug_ddi):
                raise Exception("Concat aggregator needs same length for user and item embedding")
            # 对比学习的知识表示学习
            for i in range(0, len(drug_kg)):
                # 计算损失
                kge_loss += self.RKG_Embedding_Loss(drug_kg[i], drug_kg_two[i], drug_kg, drug_kg_two)
                # 拼接
            for i in range(1, len(drug_kg)):
                e_u = torch.cat((drug_kg[i], e_u), dim=-1)
            for i in range(1, len(drug_kg_two)):
                e_p_u = torch.cat((drug_kg_two[i], e_p_u), dim=-1)
            # 目的是将每一层的嵌入向量进行对比学习和拼接，生成最终用于预测任务的知识图和药物药物网络的嵌入向量。
            e1 = torch.cat((e_u, e_p_u), dim=-1)

        ###########药物二
        if self.agg == 'concat':
            if len(drug_kg) != len(drug_ddi):
                raise Exception("Concat aggregator needs same length for user and item embedding")
            # 对比学习的知识表示学习
            for i in range(0, len(drug_kg)):
                kge_loss1 += self.RKG_Embedding_Loss(drug_ddi[i], ddi_origin_embeddings[i], drug_ddi,
                                                     ddi_origin_embeddings)
            for i in range(1, len(drug_ddi)):
                e_i = torch.cat((drug_ddi[i], e_i), dim=-1)
            for i in range(1, len(ddi_origin_embeddings)):
                e_p_i = torch.cat((ddi_origin_embeddings[i], e_p_i), dim=-1)
            # 目的是将每一层的嵌入向量进行对比学习和拼接，生成最终用于预测任务的知识图和药物药物网络的嵌入向量。
            e = torch.cat((e_p_i, e_i), dim=-1)

        embeddings_0 = F.softmax(embedding_0, dim=1)
        embeddings_1 = F.softmax(embedding_1, dim=1)
        e4 = torch.cat((e1 , embeddings_0), dim=-1)
        e5 = torch.cat((e , embeddings_1), dim=-1)
        scores = (e4 * e5).sum(dim=1)
        scores = torch.sigmoid(scores)

        return scores, kge_loss + kge_loss1
# 方法用于解析传入的参数并将其保存到对象的属性中，包括实体数量、关系数量、维度、层数以及聚合方式等信息。
    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_layer = args.n_layer
        self.agg = args.agg

# 方法用于初始化模型的权重，包括实体嵌入（entity_emb）、关系嵌入（relation_emb）以及注意力机制中的线性层（attention）的权重。
    def _init_weight(self):
        # init embedding
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)


        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)


    def _knowledge_attention(self, h_set_emb, path_emb, t_emb):
        att_weights = self.attention(torch.cat((h_set_emb, path_emb), dim=-1)).squeeze(-1)
        att_weights_norm = F.softmax(att_weights, dim=-1)
        emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_emb)
        emb_i = emb_i.sum(dim=1)
        return emb_i
    # graph contrastive learning for KGE
    def RKG_Embedding_Loss(self,current_embedding, current_two_embedding, all_embedding, all_two_embeddings):

        kg_embeddings_two = all_two_embeddings[0]# 知识图药物二嵌入的第1层
        kg_embedding_one = all_embedding[0]# 知识图药物一初始嵌入的第1层

        for i in range(1, len(all_two_embeddings)):
            kg_embeddings_two= torch.cat((all_two_embeddings[i], kg_embeddings_two), dim=0)
            #kg potential embedding 聚合前i层知识图潜在嵌入表示
        for i in range(1, len(all_embedding)):
            kg_embedding_one = torch.cat((all_embedding[i], kg_embedding_one), dim=0)
            #drug_inital_embedding 聚合前i层知识图初始嵌入表示




        norm_drug_inital = F.normalize(current_embedding)
        #当前i层药物药物网络的嵌入表示
        norm_drug_potential = F.normalize(current_two_embedding)
        #聚合前k层知识图的嵌入表示
        norm_all_drug_inital = F.normalize(kg_embeddings_two)
        norm_all_drug_potential = F.normalize(kg_embedding_one)


        #内积
        pos_score_drug_inital = torch.mul(norm_drug_inital, norm_drug_potential).sum(dim=1)
        #计算得分
        ttl_score_drug_inital = torch.matmul(norm_drug_inital, norm_all_drug_inital.transpose(0, 1))
        #通过softmax函数计算得分，通过温度参数进行缩放
        pos_score_drug_inital = torch.exp(pos_score_drug_inital / self.ssl_temp)
        ttl_score_drug_inital = torch.exp(ttl_score_drug_inital / self.ssl_temp).sum(dim=1)
        #计算损失函数
        kge_loss_drug_inital = -torch.log(pos_score_drug_inital / ttl_score_drug_inital).sum()


        #for drug_inital_embedding
        pos_score_drug_potential = torch.mul(norm_drug_potential, norm_drug_inital).sum(dim=1)
        ttl_score_drug_potential = torch.matmul(norm_drug_potential, norm_all_drug_potential.transpose(0, 1))
        pos_score_drug_potential = torch.exp(pos_score_drug_potential / self.ssl_temp)
        ttl_score_drug_potential = torch.exp(ttl_score_drug_potential / self.ssl_temp).sum(dim=1)
        kge_loss_drug_potential = -torch.log(pos_score_drug_potential / ttl_score_drug_potential).sum()

        kge_loss = self.kge_weight * (kge_loss_drug_inital + kge_loss_drug_potential)

        return kge_loss
