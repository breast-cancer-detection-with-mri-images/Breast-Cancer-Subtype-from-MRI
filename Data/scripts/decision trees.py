from ml_models_helper import *
from sklearn.tree import DecisionTreeClassifier

def start():
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2
    SEED = 2454259

    total_features = get_total_features()
    train_df, val_df, test_df = train_test_val_splits(total_features.drop(columns = ['sequence', 'patient']), TRAIN_RATIO, VAL_RATIO, random_state = SEED)
    train_x, train_y_er, train_y_pr, train_y_her, train_y_mol_subtype = train_df.drop(columns = ['ER', 'PR', 'HER2', 'Mol Subtype']), train_df['ER'], train_df['PR'], train_df['HER2'], train_df['Mol Subtype']
    val_x, val_y_er, val_y_pr, val_y_her, val_y_mol_subtype = val_df.drop(columns = ['ER', 'PR', 'HER2', 'Mol Subtype']), val_df['ER'], val_df['PR'], val_df['HER2'], val_df['Mol Subtype']
    test_x, test_y_er, test_y_pr, test_y_her, test_y_mol_subtype = test_df.drop(columns = ['ER', 'PR', 'HER2', 'Mol Subtype']), test_df['ER'], test_df['PR'], test_df['HER2'], test_df['Mol Subtype']

    scaler = Scaler()
    train_x, val_x, test_x = feature_scaling(scaler, train_x, val_x, test_x)

    classification_results = pd.DataFrame()

    tree_er = DecisionTreeClassifier(criterion = 'entropy', max_depth = 4)
    report, tree_er = get_classification_report(tree_er, train_x, train_y_er, val_x, val_y_er, test_x, test_y_er, ['ER Negative', 'ER Positive'], 'ER Subtype metrics on testing set')
    result = pd.DataFrame(report, index = [0])
    result['Algorithm'] = 'Decision Tree'
    result['Subtype'] = 'ER'
    result = result[list(result.columns[-2:]) + list(result.columns[:-2])]
    classification_results = pd.concat([classification_results, result])

    tree_pr = DecisionTreeClassifier(criterion = 'gini', max_depth = 4)
    report, tree_pr = get_classification_report(tree_pr, train_x, train_y_pr, val_x, val_y_pr, test_x, test_y_pr, ['PR Negative', 'PR Positive'], 'PR Subtype metrics on testing set')
    result = pd.DataFrame(report, index = [0])
    result['Algorithm'] = 'Decision Tree'
    result['Subtype'] = 'PR'
    result = result[list(result.columns[-2:]) + list(result.columns[:-2])]
    classification_results = pd.concat([classification_results, result])

    tree_her2 = DecisionTreeClassifier(criterion = 'gini', max_depth = 10, min_samples_split = 8)
    report, tree_her2 = get_classification_report(tree_her2, train_x, train_y_her, val_x, val_y_her, test_x, test_y_her, ['HER2 Negative', 'HER2 Positive'], 'HER2 Subtype metrics on testing set')
    result = pd.DataFrame(report, index = [0])
    result['Algorithm'] = 'Decision Tree'
    result['Subtype'] = 'HER2'
    result = result[list(result.columns[-2:]) + list(result.columns[:-2])]
    classification_results = pd.concat([classification_results, result])

    tree_mol_subtype = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, min_samples_split = 8)
    report, tree_mol_subtype = get_classification_report(tree_mol_subtype, train_x, train_y_mol_subtype, val_x, val_y_mol_subtype, test_x, test_y_mol_subtype, ['Luminal', 'ER/PR pos, HER2 pos', 'HER2', 'Triple Negative'], 'Mol Subtype metrics on testing set')
    result = pd.DataFrame(report, index = [0])
    result['Algorithm'] = 'Decision Tree'
    result['Subtype'] = 'Mol Subtype'
    result = result[list(result.columns[-2:]) + list(result.columns[:-2])]
    classification_results = pd.concat([classification_results, result])

    # save_result(classification_results)
        


if __name__ == '__main__':
    start()

