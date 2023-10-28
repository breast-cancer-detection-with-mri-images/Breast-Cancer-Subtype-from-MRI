import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler as Scaler


def get_total_features(label_path='E:/ML Project/New folder/Breast-Cancer-Subtype-from-MRI/Data/Patient class labels.csv', feature_path='E:/ML Project/New folder/Breast-Cancer-Subtype-from-MRI/Data/pyradiomics_extraction.csv', sequence='pre'):
    labels = pd.read_csv(label_path)
    features = pd.read_csv(feature_path)
    pre_features = features[features['sequence'] == sequence]
    total_features = pd.merge(pre_features, labels, left_on='patient', right_on='Patient ID').drop(columns='Patient ID')

    return total_features


def train_test_val_splits(df, train_ratio = 0.8, val_ratio = 0.2, random_state = 2454259):
   
    val_ratio_adj = val_ratio / (1-train_ratio)

    train_df, val_df = train_test_split(df, train_size = train_ratio, random_state= random_state)
    val_df, test_df = train_test_split(val_df, train_size = val_ratio_adj, random_state= random_state)

    print(len(train_df))

    return train_df, val_df, test_df


def train_pipeline(model, trainx, trainy, valx, valy):
    model.fit(trainx, trainy)
    preds = model.predict(trainx)
    preds_val = model.predict(valx)

    acc_train, acc_val = accuracy_score(trainy, preds), accuracy_score(valy, preds_val)
    prec_train, prec_val = precision_score(trainy, preds, average = 'weighted'), precision_score(valy, preds_val, average = 'weighted')
    rec_train, rec_val = recall_score(trainy, preds, average = 'weighted'), recall_score(valy, preds_val, average = 'weighted')
    f1_train, f1_val = f1_score(trainy, preds, average = 'weighted'), f1_score(valy, preds_val, average = 'weighted')

    metrics = {'train_acc' : acc_train, 'val_acc' : acc_val,
               'train_prec': prec_train, 'val_prec': prec_val,
               'train_rec' : rec_train,  'val_rec' : rec_val,
               'train_f1' : f1_train, 'val_f1': f1_val}


    print("Training Accuracy: {:.4f}, Validation Accuracy: {:.4f}".format(acc_train, acc_val))
    print("Training Precision: {:.4f}, Validation Precision: {:.4f}".format(prec_train, prec_val))
    print("Training Recall: {:.4f}, Validation Recall: {:.4f}".format(rec_train, rec_val))
    print("Training F1-Score: {:.4f}, Validation F1-Score: {:.4f}".format(f1_train, f1_val))
    print()

    return model, metrics


def get_classification_report(model, trainx, trainy, valx, valy, testx, testy, class_names, title, subtype = None):
    model, metrics = train_pipeline(model, trainx, trainy, valx, valy)
    pred = model.predict(testx)


    acc_test = accuracy_score(testy, pred)
    prec_test = precision_score(testy, pred, average = 'weighted')
    rec_test = recall_score(testy, pred, average = 'weighted')
    f1_test = f1_score(testy, pred, average = 'weighted')
    
    metrics = {}
    metrics.update(
            {'test_acc' : acc_test,
            'test_prec': prec_test,
            'test_rec' : rec_test,
            'test_f1' : f1_test})

    if len(trainy.unique()) == 2:
        probs = model.predict_proba(testx)
        aucroc = roc_auc_score(testy, probs[:, 1], average = 'weighted')
        metrics.update({'test_aucroc' : aucroc})

    # cls_report = classification_report(testy, pred, target_names = class_names, output_dict = True)
    # sns.heatmap(pd.DataFrame(cls_report).iloc[:-1, :].T, annot=True)
    # plt.title(title)
    return metrics, model


def feature_scaling(scaler, train_x, val_x, test_x):
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    val_x = scaler.transform(val_x)
    test_x = scaler.transform(test_x)

    return train_x, val_x, test_x


def save_result(classification_results):
    try:
        with open("Data/scripts/Result.csv") as f:
            first = f.readline()
            # if first == 'Algorithm,Subtype,train_acc,val_acc,train_prec,val_prec,train_rec,val_rec,train_f1,val_f1,test_acc,test_prec,test_rec,test_f1,test_aucroc\n':
            if len(first) > 0:
                classification_results.to_csv("Data/scripts/Result.csv", mode='a', header=False, index=False) 
            else:
                classification_results.to_csv("Data/scripts/Result.csv", mode='a', index=False)
    except:
        classification_results.to_csv("Data/scripts/Result.csv", index=False)
