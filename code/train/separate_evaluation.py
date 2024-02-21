from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from tensorflow.keras.layers import Input, Attention, LayerNormalization, Dense
from sklearn.preprocessing import StandardScaler



pd.set_option('display.float_format', '{:.10f}'.format)
#读取特征

df_seq_feature = pd.read_csv('../../feature/miRNA_seq_feature_64.csv', index_col='Index')
df_mRNA_feature = pd.read_csv('../../feature/miRNA_mRNA_feature.csv',header=None)
df_drug_feature = pd.read_csv('../../feature/gate_feature_drug_0.8_128_0.01.csv',header=None)
df_dis_feature = pd.read_csv('../../feature/gate_feature_disease_0.8_128_0.01.csv',header=None)
df_loc = pd.read_csv('../../datasets/miRNA_localization.csv',header=None)


dis_feature = df_dis_feature.values
seq_feature = df_seq_feature.values
mRNA_feature = df_mRNA_feature.values
drug_feature = df_drug_feature.values
miRNA_loc = df_loc.values

print(miRNA_loc.shape)

# 将特征concatenate起来
merge_feature = np.concatenate((seq_feature, dis_feature ,drug_feature, mRNA_feature), axis=1)
print(merge_feature.shape)


#多标签分类模型
def create_multi_label_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # 自注意力
    attention = tf.keras.layers.Attention()([inputs, inputs])
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention)

    # 全连接网络
    flatten = tf.keras.layers.Flatten()(attention)
    dense_1 = Dense(64, activation='relu')(flatten)
    dense_2 = Dense(32, activation='relu')(dense_1)

    # 输出层改为多标签分类
    outputs = Dense(num_classes, activation='sigmoid')(dense_2)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

n_splits = 10

# 标准化特征数据
scaler = StandardScaler()
merge_feature_scaled = scaler.fit_transform(merge_feature)



# 定义类别数量
num_classes = 7

# 随机种子以保持可复现性
random_seed = 42


# 手动实现十折交叉验证
fold_size = len(merge_feature_scaled) // n_splits

# 设置随机种子以确保可复现性
np.random.seed(random_seed)



class_name = ['Cytoplasm', 'Exosome', 'Nucleolus', 'Nucleus', 'Extracellular vesicle', 'Microvesicle', 'Mitochondrion']

auc_ls = []
aupr_ls = []
for iteration in range(10):
    all_true_labels = []
    all_predicted_labels = []
    for i in range(n_splits):
        # 打乱数据集
        X, y = shuffle(merge_feature_scaled, miRNA_loc, random_state=random_seed)

        # 划分训练集和测试集
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        test_indices = range(test_start, test_end)
        train_indices = [j for j in range(len(X)) if j not in test_indices]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        model = create_multi_label_model(input_shape=(324,), num_classes=num_classes)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=10, batch_size=32)

        y_pred = model.predict(X_test)
        print(y_pred.shape)

        # 将真实标签和预测标签添加到累积变量中
        all_true_labels.append(y_test)
        all_predicted_labels.append(y_pred)

    # 合并所有折的结果
    all_true_labels = np.vstack(all_true_labels)
    print(all_true_labels.shape)
    all_predicted_labels = np.vstack(all_predicted_labels)
    print(all_predicted_labels.shape)


    avg_AUC = 0
    avg_AUPR = 0
    result = []
    for class_idx in range(num_classes):
        true_labels_class = all_true_labels[:, class_idx]
        predicted_labels_class = all_predicted_labels[:, class_idx]

        accuracy = accuracy_score(true_labels_class, (predicted_labels_class > 0.5).astype(int))
        precision = precision_score(true_labels_class, (predicted_labels_class > 0.5).astype(int))
        recall = recall_score(true_labels_class, (predicted_labels_class > 0.5).astype(int))
        f1 = f1_score(true_labels_class, (predicted_labels_class > 0.5).astype(int))
        auc_score = roc_auc_score(true_labels_class, predicted_labels_class)
        aupr_score = average_precision_score(true_labels_class, predicted_labels_class)

        avg_AUC += auc_score
        avg_AUPR += aupr_score

        print(f"Iteration {iteration}, Class {class_name[class_idx]} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC: {auc_score}, AUPR: {aupr_score}")
    avg_AUPR /= num_classes
    avg_AUC /= num_classes
    print(f"Iteration {iteration} - AVG AUC: {avg_AUC}")
    print(f"Iteration {iteration} - AVG AUPR: {avg_AUPR}")
    auc_ls.append(avg_AUC)
    aupr_ls.append(avg_AUPR)

print(f'avg_AUC:{sum(auc_ls)/len(auc_ls)}')
print(f'avg_AUPR:{sum(aupr_ls)/len(aupr_ls)}')
