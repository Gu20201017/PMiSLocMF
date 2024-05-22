import csv
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec


if __name__ == '__main__':
    df_ms = pd.read_csv('../../datasets/miRNA_seq_sim.csv',header=None)
    ms = df_ms.values
    print(ms.shape)

    # 创建一个有权重的图
    G = nx.Graph()

    # 添加 miRNA 节点
    for i in range(1041):
        G.add_node(i)

    # 添加边并设置边的权重
    for i in range(1041):
        for j in range(i+1, 1041):
            similarity = ms[i, j]
            if similarity > 0:
                G.add_edge(i, j, weight=similarity)

    # 创建 Node2Vec 对象
    node2vec = Node2Vec(G, dimensions=64, walk_length=150, num_walks=200, workers=1, weight_key='weight')

    # 训练模型
    model = node2vec.fit()

    # 提取 miRNA 特征向量
    ms_features = {str(node): model.wv[str(node)] for node in G.nodes()}
    print(ms_features)

    # 将字典转换为 DataFrame
    df = pd.DataFrame.from_dict(ms_features, orient='index')

    # 重命名 DataFrame 的列名，将每一维作为一列
    df.columns = [f'Dimension_{i}' for i in range(64)]  # 列名可以根据需要自定义

    # 添加索引名称
    df.index.name = 'Index'

    df.to_csv('../../feature/miRNA_seq_feature_64.csv')