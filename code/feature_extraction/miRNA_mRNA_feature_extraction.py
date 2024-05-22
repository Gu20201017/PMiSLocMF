import csv
import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec


if __name__ == '__main__':
    df_miRNA_mRNA = pd.read_csv('../../datasets/miRNA_mRNA_matrix.txt',header=None,delimiter='\t')
    miRNA_mRNA_matrix = df_miRNA_mRNA.values

    df_mRNA_loc = pd.read_csv('../../datasets/mRNA_localization.txt', header=None, delimiter='\t')
    mRNA_loc = df_mRNA_loc.values

    miRNA_mRNA_ratio_vector = []

    for i in range(miRNA_mRNA_matrix.shape[0]):
        temp = [0,0,0,0]
        for j in range(miRNA_mRNA_matrix.shape[1]):
            if miRNA_mRNA_matrix[i][j] == 1:
                #找对应j行的mRNA的定位情况
                for k in range(4):
                    if mRNA_loc[j][k] == 1:
                        temp[k] += 1
        #处理完了一个miRNA
        sum_number = sum(temp)
        if sum_number != 0:
            for k in range(len(temp)):
                temp[k] = temp[k] / sum_number
        miRNA_mRNA_ratio_vector.append(temp)

    print(miRNA_mRNA_ratio_vector)
    # 指定要写入的CSV文件名
    csv_file = "../../feature/miRNA_mRNA_feature.csv"
    # 写入CSV文件
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in miRNA_mRNA_ratio_vector:
            writer.writerow(row)
