"""
This code is implemented for HW2 at Tsinghua SIGS
Using logistic regression for binary classification
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import  pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np


# perform data normalization
def normalize(data, mode="0"):
    if mode == "0":
        data_mean, data_std = data.mean(0), data.std(0)
        new_data = (data - data_mean) / (data_std+1e-5)
        return new_data
    else:
        return (data - data.min(0))/(data.max(0)+1e-5)

# draw roc curve
def draw_roc():
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='blue', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Sensitivity)')
    plt.ylabel('True Positive Rate (1-Specificity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# note for logistic regression, the bias/intercept is just a number
def analyze_saliency():
    weights, bias = model.coef_, model.intercept_
    weights = np.abs(weights)[0]

    print((np.where(weights > np.mean(weights)))[0])
    print(np.argmax(weights))


## load the dataset and transform the format to numpy
if __name__ == '__main__':
    data_path = "./data1/" # input your data path
    train_data1 = pd.read_csv(data_path + 'train1_icu_data.csv')
    train_data2 = pd.read_csv(data_path + 'train2_icu_data.csv')
    train_label1 = pd.read_csv(data_path + 'train1_icu_label.csv')
    test_data1 = pd.read_csv(data_path + 'test1_icu_data.csv')
    test_label1 = pd.read_csv(data_path + 'test1_icu_label.csv')

    num_sample = train_data1.shape[0]
    train_data1, train_label1 = train_data1.to_numpy(), train_label1.to_numpy() # 5000x108
    test_data1, test_label1 = test_data1.to_numpy(), test_label1.to_numpy() # 1097x108
    train_label1, test_label1 = train_label1.ravel(), test_label1.ravel()

    # build an instance of logistic regression

    num_fold = 5
    # training and 5-fold cross-validation
    division = num_sample // num_fold
    all_index = list(range(0, num_sample))
    normalization_mode = "1"
    for i in range(num_fold):
        sample_index_validation = all_index[division*i:division*(i+1)]
        sample_index_train = all_index[:division*i] + all_index[division*(i+1):]
        X_train, X_valid = normalize(train_data1[sample_index_train], mode=normalization_mode), normalize(train_data1[sample_index_validation], mode=normalization_mode)
        y_train, y_valid = train_label1[sample_index_train], train_label1[sample_index_validation]
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred)
        y_pred = model.predict(X_valid)
        valid_accuracy = accuracy_score(y_valid, y_pred)
        print("{}-th fold with train accuracy {:.2f}, valid accuracy: {:.2f}".format(i, train_accuracy, valid_accuracy))

    model = LogisticRegression()
    X_train = normalize(train_data1, mode=normalization_mode)
    model.fit(X_train, train_label1)
    y_test = test_label1
    X_test = normalize(test_data1, mode=normalization_mode)
    y_pred = model.predict(X_test)
    weight, intercept = model.coef_, model.intercept_
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2f}")
    draw_roc()


    ######################## the following code is written for saliency analysis of features#########################
    analyze_saliency()

    import statsmodels.api as sm

    X_train, y_train = normalize(train_data1, "0"), train_label1.ravel()
    X_test, y_test = normalize(test_data1, "0"), test_label1.ravel()
    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)
    logit_model = sm.Logit(y_train, X_train_sm)
    result = logit_model.fit()
    y_pred_prob = result.predict(X_test_sm)  # 返回的是概率值
    y_pred = np.where(y_pred_prob >= 0.5, 1, 0)  # 通过设定阈值 0.5 进行分类

    p_values = result.pvalues
    p_values_series = pd.Series(p_values)
    p_values_cleaned = p_values_series.fillna(99)
    p_values = p_values_cleaned.to_numpy().ravel()
    salient_index = np.where(p_values<0.5)[0]
    min_value, min_index = np.min(p_values), np.argmin(p_values)
    print(salient_index, min_value, min_index)


