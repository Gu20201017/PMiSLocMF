import csv
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

if __name__ == '__main__':
    df_miRNA_dis = pd.read_csv('../../../datasets/miRNA_disease.csv', header=None)
    miRNA_dis_matrix = df_miRNA_dis.values
    print(miRNA_dis_matrix.shape)

    # 创建一个空图
    G = nx.Graph()

    # 添加miRNA节点
    for i in range(1041):
        G.add_node(f"miRNA_{i}")

    # 添加疾病节点
    for i in range(640):
        G.add_node(f"disease_{i}")

    # 遍历矩阵，添加边
    for miRNA_index in range(1041):
        for disease_index in range(640):
            interaction = miRNA_dis_matrix[miRNA_index, disease_index]
            if interaction != 0:
                G.add_edge(f"miRNA_{miRNA_index}", f"disease_{disease_index}")

    # 创建 Node2Vec 对象
    node2vec = Node2Vec(G, dimensions=128, walk_length=150, num_walks=200, workers=1)

    # 训练模型
    model = node2vec.fit()

    # 提取 miRNA 特征向量
    miRNA_features = {}
    for miRNA_node in range(1041):
        miRNA_node_str = f"miRNA_{miRNA_node}"
        miRNA_features[miRNA_node_str] = model.wv[miRNA_node_str]

    df = pd.DataFrame.from_dict(miRNA_features, orient='index')
    df.columns = [f'Dimension_{i}' for i in range(128)]  # 列名可以根据需要自定义
    print(df)
    df.to_csv('../../feature/miRNA_disease_feature_128.csv')