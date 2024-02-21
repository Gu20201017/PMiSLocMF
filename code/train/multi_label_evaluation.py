from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from tensorflow.keras.layers import Input, Attention, LayerNormalization, Dense
from sklearn.preprocessing import StandardScaler


def multi_evaluate(y_pred, y_test, M):
    N = len(y_test)
    Aiming = 0
    Coverage = 0
    Accuracy = 0
    Absolute_True = 0
    Absolute_False = 0
    for i in range(N):
        union_set_len = np.sum(np.maximum(y_pred[i], y_test[i]))
        inter_set_len = np.sum(np.minimum(y_pred[i], y_test[i]))
        y_pred_len = np.sum(y_pred[i])
        y_test_len = np.sum(y_test[i])
        Aiming += inter_set_len / y_pred_len
        Coverage += inter_set_len / y_test_len
        Accuracy += inter_set_len / union_set_len
        Absolute_True += int(np.array_equal(y_pred[i], y_test[i]))
        Absolute_False += (union_set_len - inter_set_len) / M
    Aiming = Aiming / N
    Coverage = Coverage / N
    Accuracy = Accuracy / N
    Absolute_True = Absolute_True / N
    Absolute_False = Absolute_False / N
    return Aiming, Coverage, Accuracy, Absolute_True, Absolute_False


#读取有位置信息的索引
df_loc_index = pd.read_csv('../../datasets/miRNA_name_1041_have_loc_information_index.txt',header=None)
loc_index = df_loc_index[0].tolist()
select_row = np.array([value == 1 for value in loc_index])
print(select_row)


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

miRNA_loc = miRNA_loc[select_row]


# 将特征concatenate起来
merge_feature = np.concatenate((seq_feature,dis_feature,drug_feature,mRNA_feature), axis=1)
merge_feature = merge_feature[select_row]
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
print(merge_feature_scaled.shape)


# 定义类别数量
num_classes = 7


# 随机种子以保持可复现性
random_seed = 42


# 手动实现十折交叉验证
fold_size = len(merge_feature_scaled) // n_splits

# 设置随机种子以确保可复现性
np.random.seed(random_seed)


all_Aiming = []
all_Coverage = []
all_Accuracy = []
all_Absolute_True = []
all_Absolute_False = []
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

    y_pred_label = (y_pred > 0.5).astype(int)
    print(y_pred_label.shape)
    print(y_test.shape)


    Aiming, Coverage, Accuracy, Absolute_True, Absolute_False = multi_evaluate(y_pred_label, y_test, 7)
    print(f"Aiming:{Aiming},Coverage:{Coverage}, Accuracy:{Accuracy}, Absolute_True:{Absolute_True}, Absolute_False:{Absolute_False}")
    all_Aiming.append(Aiming)
    all_Coverage.append(Coverage)
    all_Accuracy.append(Accuracy)
    all_Absolute_True.append(Absolute_True)
    all_Absolute_False.append(Absolute_False)

print(f"avg_Aiming:{sum(all_Aiming) / len(all_Aiming)},avg_Coverage:{sum(all_Coverage) / len(all_Coverage)}, avg_Accuracy:{sum(all_Accuracy) / len(all_Accuracy)}, avg_Absolute_True:{sum(all_Absolute_True) / len(all_Absolute_True)}, avg_Absolute_False:{sum(all_Absolute_False) / len(all_Absolute_False)}")





