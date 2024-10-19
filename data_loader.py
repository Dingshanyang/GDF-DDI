
import collections
import os
import numpy as np
import logging
from random import sample
import pickle



logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)

# 函数是整个数据加载和预处理流程的入口函数。它负责调用其他函数来加载评分数据、知识图谱数据，并进行协同传播，构建不同类型的实体集合和三元组集合。







def load_drug_data(filename):
    drug_data = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 3:
                drug_id = int(parts[0])
                smiles = parts[2]
                drug_data[drug_id] = smiles
    return drug_data




# 加载药物数据


def load_data(args):
    logging.info("================== preparing data ===================")
    train_data,kg_init_entity_sets, ddi_potential_entity_setss = load_rating(args)
    n_entity, n_relation,DDI_index = load_reanden(args)
    kg = load_kg(args)

    logging.info("contructing kg initial kg triple sets ...")
    #triple_sets = kg_propagation(args, kg, kg_init_entity_sets, args.kg_triple_set_size)
    # 将结果存储到文件中
    #with open('triple_sets.pkl', 'wb') as f:
        #pickle.dump(triple_sets, f)
    # 在需要的时候从文件中读取数据
    with open('triple_sets.pkl', 'rb') as f:
        triple_sets = pickle.load(f)
    kg_init_entity_set = triple_sets
    logging.info("constructing kg potential kg triple sets ... ")
    kg_triple_sets = triple_sets
    logging.info("contructing ddi potential triple sets ...")
    #triple_set = kg_propagation(args, DDI_index, kg_init_entity_sets, args.kg_triple_set_size)
    #with open('triple_set.pkl', 'wb') as f:
        #pickle.dump(triple_set, f)
    with open('triple_set.pkl', 'rb') as f:
        triple_set = pickle.load(f)
    DDI_potential_entity_set = triple_set
    logging.info("constructing ddi origin kg triple sets ... ")
    DDI_origin_triple_sets = triple_set




    ####################第二个药物信息
    logging.info("constructing kg_init_entity_set1 ... ")
    #triple_1 = kg_propagation(args, kg, ddi_potential_entity_setss, args.kg_triple_set_size)
    # 将结果存储到文件中
    #with open('triple_1.pkl', 'wb') as f:
        #pickle.dump(triple_1, f)
    # 在需要的时候从文件中读取数据
    with open('triple_1.pkl', 'rb') as f:
        triple_1 = pickle.load(f)
    kg_drug2_triple = triple_1
    kg_drug2_triple2 = triple_1
    logging.info("constructing ddi_potential_entity_set1 ... ")
    #triple_2 = kg_propagation(args, DDI_index, ddi_potential_entity_setss, args.kg_triple_set_size)
    # 将结果存储到文件中
    #with open('triple_2.pkl', 'wb') as f:
        #pickle.dump(triple_2, f)
    # 在需要的时候从文件中读取数据
    with open('triple_2.pkl', 'rb') as f:
        triple_2 = pickle.load(f)
    DDI_drug2_triple = triple_2
    DDI_drug2_triple2 = triple_2
    smiles = '../data/music/drug_data.txt'
    id_smiles = load_drug_data(smiles)
    return train_data,  n_entity, n_relation, kg_init_entity_set, DDI_potential_entity_set, kg_triple_sets, DDI_origin_triple_sets,kg_drug2_triple,DDI_drug2_triple, kg_drug2_triple2,DDI_drug2_triple2,id_smiles









# 函数用于加载评分数据。它首先确定评分文件的路径，然后检查是否已经存在以.npy格式保存的评分数据，如果不存在则从.txt文件中加载评分数据并保存为.npy格式。
def load_rating(args):
    file_path = f'../data/kegg/approved_example.npy'
    rating_np = np.load(file_path)
    return dataset_split(rating_np)


def load_reanden(args):
    rating = '../data/kegg/approved_example.txt'
    approved = open(rating, 'r')
    approved_lines = approved.readlines()
    drug_set = set()
    for i in approved_lines:
        line = i.strip().split('\t')
        drug_set.add(line[0])
    rating1 = '../data/' + args.dataset + '/entity2id.txt'
    entity_file = open(rating1, 'r')
    entity_lines = entity_file.readlines()
    entity_dict = dict()
    flag = 1
    for en in entity_lines:
        if flag == 1:
            flag = 0
            continue
        en_line = en.strip().split('\t')
        entity_dict[en_line[1]] = en_line[0]
    rating2 = '../data/' + args.dataset + '/relation2id.txt'
    relation_file = open(rating2, 'r')
    relation_lines = relation_file.readlines()
    relation_dict = dict()
    flag = 1
    for re in relation_lines:
        if flag == 1:
            flag = 0
            continue
        re_line = re.strip().split('\t')
        relation_dict[re_line[1]] = re_line[0]
    n_relation = len(relation_dict)
    n_entity = len(entity_dict)
    DDI_index = '../data/music/approved_example'
    logging.info("loading kg file: %s.npy", DDI_index)
    if os.path.exists(DDI_index + '.npy'):
        DDI_index_np = np.load(DDI_index + '.npy')
    else:
        DDI_index_np = np.loadtxt(DDI_index + '.txt', dtype=np.int32)
        np.save(DDI_index + '.npy', DDI_index_np)
    DDI_index = construct_DDI(DDI_index_np)
    return n_entity, n_relation, DDI_index




def kg_propagation(args, kg, init_entity_set, set_size):
    # triple_sets: [n_obj][n_layer](h,r,t)x[set_size]
    triple_sets = collections.defaultdict(list)
    for obj in init_entity_set.keys():
        if args.n_layer == 0:
            n_layer = 1
        else:
            n_layer = args.n_layer
        for l in range(n_layer):
            h, r, t = [], [], []
            if l == 0:
                entities = init_entity_set[obj]
            else:
                entities = triple_sets[obj][-1][2]
            for entity in entities:
                for tail_and_relation in kg[entity]:
                    h.append(entity)
                    t.append(tail_and_relation[0])
                    r.append(tail_and_relation[1])
            if len(h) == 0:
                triple_sets[obj].append(triple_sets[obj][-1])
            else:
                indices = np.random.choice(len(h), size=set_size, replace=(len(h) < set_size))
                h = [h[i] for i in indices]
                r = [r[i] for i in indices]
                t = [t[i] for i in indices]
                triple_sets[obj].append((h, r, t))
    return triple_sets




def dataset_split(rating_np):
    n_ratings = rating_np.shape[0]
    train_indices = np.random.choice(n_ratings, size=int(n_ratings ), replace=False)
    kg_init_entity_set, ddi_potential_entity_set = collaboration_propagation(rating_np, train_indices)
    train_indices = [i for i in train_indices if rating_np[i][0] in kg_init_entity_set.keys()]
    train_data = rating_np[train_indices]
    return train_data, kg_init_entity_set, ddi_potential_entity_set


# load_kg 函数负责加载知识图谱数据，它接受一个参数：
# args：包含了数据集名称等信息的对象。
# 在函数中，首先根据传入的参数构建知识图谱文件的路径 kg_file。然后，通过判断文件是否存在，分别使用 np.load 和 np.loadtxt 加载知识图谱数据。如果是第一次加载，则将加载后的数据保存为 .npy 文件，以便下次直接加载。
# 接着，计算知识图谱的实体数量 n_entity 和关系数量 n_relation，并调用 construct_kg 函数构建知识图谱。
# 最后，将实体数量、关系数量和构建好的知识图谱返回。
def load_kg(args):
    kg_file = '../data/kegg/train2id'
    logging.info("loading kg file: %s.npy", kg_file)
    with open(kg_file + '.txt', 'r') as file:
        lines = file.readlines()[1:]  # 跳过第一行
        kg_np = np.loadtxt(lines, dtype=np.int32)
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        np.save(kg_file + '.npy', kg_np)
    kg = construct_kg(kg_np)
    return  kg
"""
def load_DDI(args):
    DDI_index = '../data/' + args.dataset + '/approved_example'
    logging.info("loading kg file: %s.npy", DDI_index)
    if os.path.exists(DDI_index + '.npy'):
        DDI_index_np = np.load(DDI_index + '.npy')
    else:
        DDI_index_np = np.loadtxt(DDI_index + '.txt', dtype=np.int32)
        np.save(DDI_index + '.npy', DDI_index_np)
    n_entity = len(set(DDI_index_np[:, 0]) )
    n_relation = len(set(DDI_index_np[:, 2]))
    DDI_index = construct_DDI(DDI_index_np)
    return n_entity,n_relation,DDI_index
"""


def construct_kg(kg_np):
    logging.info("constructing knowledge graph ...")
    kg_np = kg_np[1:]
    kg = collections.defaultdict(list)
    # for head, relation, tail in kg_np:
    #     kg[head].append((tail, relation))
    for triple in kg_np:
        head = triple[0]
        relation = triple[2]
        tail = triple[1]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg



def construct_DDI(DDI_index_np):
    logging.info("constructing DDI graph ...")
    DDI = collections.defaultdict(list)
    # for head, relation, tail in kg_np:
    #     kg[head].append((tail, relation))
    for triple in DDI_index_np:
        head = triple[0]
        relation = triple[2]
        tail = triple[1]
        # treat the KG as an undirected graph
        if head not in DDI:
            DDI[head] = []
        DDI[head].append((tail, relation))
        if tail not in DDI:
            DDI[tail] = []
        DDI[tail].append((head, relation))
    return DDI



def collaboration_propagation(rating_np, train_indices):
    kg_history_item_dict = dict()
    ddi_history_user_dict = dict()
    logging.info("contructing kg  initial entity set ...")
    for i in train_indices:
        head = rating_np[i][0]
        tail = rating_np[i][1]
        rating = rating_np[i][2]
        if rating != 2:
            if head not in kg_history_item_dict:
                kg_history_item_dict[head] = []
            kg_history_item_dict[head].append(tail)

            if tail not in ddi_history_user_dict:
                ddi_history_user_dict[tail] = []
            ddi_history_user_dict[tail].append(head)
    return kg_history_item_dict,ddi_history_user_dict
