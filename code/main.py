from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from tensorflow.keras.layers import Input, Attention, LayerNormalization, Dense
from sklearn.preprocessing import StandardScaler

def create_multi_label_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    attention = tf.keras.layers.Attention()([inputs, inputs])
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention)

    flatten = tf.keras.layers.Flatten()(attention)
    dense_1 = Dense(64, activation='relu')(flatten)
    dense_2 = Dense(32, activation='relu')(dense_1)

    outputs = Dense(num_classes, activation='sigmoid')(dense_2)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def multi_evaluate(y_pred, y_test, M):
    N = len(y_test)
    count_Aiming = 0
    count_Coverage = 0
    count_Accuracy = 0
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
        if y_pred_len > 0:
            Aiming += inter_set_len / y_pred_len
            count_Aiming = count_Aiming + 1
        if y_test_len > 0:
            Coverage += inter_set_len / y_test_len
            count_Coverage = count_Coverage + 1
        if union_set_len > 0:
            Accuracy += inter_set_len / union_set_len
            count_Accuracy = count_Accuracy + 1
        Absolute_True += int(np.array_equal(y_pred[i], y_test[i]))
        Absolute_False += (union_set_len - inter_set_len) / M
    Aiming = Aiming / count_Aiming
    Coverage = Coverage / count_Coverage
    Accuracy = Accuracy / count_Accuracy
    Absolute_True = Absolute_True / N
    Absolute_False = Absolute_False / N
    return Aiming, Coverage, Accuracy, Absolute_True, Absolute_False

df_seq_feature = pd.read_csv('../feature/miRNA_seq_feature_64.csv', index_col='Index')
df_mRNA_feature = pd.read_csv('../feature/miRNA_mRNA_feature.csv',header=None)
df_drug_feature = pd.read_csv('../feature/gate_feature_drug_0.8_128_0.01.csv',header=None)
df_dis_feature = pd.read_csv('../feature/gate_feature_disease_0.8_128_0.01.csv',header=None)
df_loc = pd.read_csv('../datasets/miRNA_localization.csv',header=None)
df_loc_index = pd.read_csv('../datasets/miRNA_have_loc_information_index.txt', header=None)
loc_index = df_loc_index[0].tolist()
select_row = np.array([value == 1 for value in loc_index])

dis_feature = df_dis_feature.values
seq_feature = df_seq_feature.values
mRNA_feature = df_mRNA_feature.values
drug_feature = df_drug_feature.values
miRNA_loc = df_loc.values


merge_feature = np.concatenate((seq_feature, dis_feature, drug_feature, mRNA_feature), axis=1)

n_splits = 10

scaler = StandardScaler()
merge_feature_scaled = scaler.fit_transform(merge_feature)
miRNA_loc_multilabel = miRNA_loc[select_row]
merge_feature_scaled_multilabel = merge_feature_scaled[select_row]

num_classes = 7
random_seed = 42
fold_size = len(merge_feature_scaled) // n_splits
np.random.seed(random_seed)
class_name = ['Cytoplasm', 'Exosome', 'Nucleolus', 'Nucleus', 'Extracellular vesicle', 'Microvesicle', 'Mitochondrion']

auc_ls = [0] * 7
aupr_ls = [0] * 7
for i in range(n_splits):
    X, y = shuffle(merge_feature_scaled, miRNA_loc, random_state=random_seed)

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

    for class_idx in range(num_classes):
        true_label = y_test[:, class_idx]
        pre_label = y_pred[:, class_idx]

        accuracy = accuracy_score(true_label, (pre_label > 0.5).astype(int))
        precision = precision_score(true_label, (pre_label > 0.5).astype(int))
        recall = recall_score(true_label, (pre_label > 0.5).astype(int))
        f1 = f1_score(true_label, (pre_label > 0.5).astype(int))
        auc_score = roc_auc_score(true_label, pre_label)
        aupr_score = average_precision_score(true_label, pre_label)

        print(f"Class {class_name[class_idx]} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC: {auc_score}, AUPR: {aupr_score}")
        auc_ls[class_idx] += auc_score
        aupr_ls[class_idx] += aupr_score

fold_size_multilabel = len(merge_feature_scaled_multilabel) // n_splits
all_Aiming = []
all_Coverage = []
all_Accuracy = []
all_Absolute_True = []
all_Absolute_False = []
for i in range(n_splits):
    X, y = shuffle(merge_feature_scaled_multilabel, miRNA_loc_multilabel, random_state=random_seed)

    test_start = i * fold_size_multilabel
    test_end = (i + 1) * fold_size_multilabel
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

avg_auc = 0
avg_aupr = 0
print("-----------------------Final Result-----------------------")
for class_idx in range(num_classes):
    print(f"Class {class_name[class_idx]} - AUC: {auc_ls[class_idx] / 10}, AUPR: {aupr_ls[class_idx] / 10}")
    avg_auc += auc_ls[class_idx] / 10
    avg_aupr += aupr_ls[class_idx] / 10
avg_auc /= 7
avg_aupr /= 7
print(f"avg_AUC: {avg_auc}, avg_AUPR: {avg_aupr}")
print(f"avg_Aiming:{sum(all_Aiming) / len(all_Aiming)},avg_Coverage:{sum(all_Coverage) / len(all_Coverage)}, avg_Accuracy:{sum(all_Accuracy) / len(all_Accuracy)}, avg_Absolute_True:{sum(all_Absolute_True) / len(all_Absolute_True)}, avg_Absolute_False:{sum(all_Absolute_False) / len(all_Absolute_False)}")






