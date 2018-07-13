import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
import pandas as pd

def def_rates_by_categorical(df, 
                             column, 
                             with_variance=False, 
                             sort=True):
    """"""
    grouped = df.groupby([column,'loan_status'])
    def_counts = grouped['loan_amnt'].count().unstack()
    N = def_counts.sum(axis=1)
    props = def_counts['Charged Off'] / N
    if sort:
        props = props.sort_values()
    var = ((props * (1 - props)) / N) ** (.5)
    if with_variance:
        ax = props.plot(kind = 'bar', yerr = var)
    else:
        ax = props.plot(kind = 'bar')
    ax.set_ylabel("Default Rate (0-1)")
    ax.set_title("Default Rates by {}".format(column))
    ax.set_xlabel(column)
    return ax
    
def round_to_nearest(x, base=1):
    return base * int(x / base)

def plot_roc_curve(y_test, X_test, model_dict):
    y_test_ = label_binarize(y_test, classes=[0, 1, 2])[:, :2]
    preds = {}
    fpr = {}
    tpr = {}
    roc_auc = {}
    f, ax = plt.subplots(1)
    
    #plt.figure()
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    
    plot_data = {}
    
    for model_key in model_dict:
        preds = model_dict[model_key].predict_proba(X_test)
        fpr = {}
        tpr = {} 
        roc_auc = {}        
        
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(y_test_[:, i], preds[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_.ravel(), preds.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        name = "%s: (AUC = %0.2f)" % (model_key, roc_auc[1])
        plot_data = pd.DataFrame(tpr[1], index=fpr[1], columns = [name])
        plot_data.plot(ax=ax)
    plt.show()
    return ax


def classifaction_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = re.split('[ ]+', line)
        row['class'] = row_data[1]
        row['precision'] = float(row_data[2])
        row['recall'] = float(row_data[3])
        row['f1_score'] = float(row_data[4])
        row['support'] = float(row_data[5])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    return dataframe

def hover(hover_color="#ffff99"):
    return dict(selector="tr:hover",
                props=[("background-color", "%s" % hover_color)])

styles = [
    hover(),
    dict(selector="th", props=[("font-size", "100%"),
                               ("text-align", "center")]),
    dict(selector="caption", props=[("caption-side", "bottom")])
]

from sklearn.metrics import confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    

def plot_discriminative_thresholds(model, y_true, include_f1=True):
    probabilities = model.predict_proba(X_test)
    prec = {}
    rec = {}
    f1_scores = {}
    for i in map(lambda x: x / 20., range(20)):
        predictions = np.apply_along_axis(lambda x: x[1] > i, 1, probabilities)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, predictions)
        prec[i] = precision[1]
        rec[i] = recall[1]
        f1_scores[i] = f1[1]
        
    best_i = pd.Series(f1_scores).idxmax()


    default_thresholds = pd.DataFrame({'recall':pd.Series(rec), 
                                       'precision':pd.Series(prec),
                                       'f1':pd.Series(f1_scores)})
    if include_f1:
        ax = default_thresholds.plot()
    else:
        ax = default_thresholds[['recall','precision']].plot()
    ax.axvline(best_i, linestyle= "--", color = 'gray')
    ax.text(1.1, .6, "F1 Optimal Threshold: %.2f" % best_i)
    ax.text(1.1, .52, "Optimal Recall: %.2f" % rec[best_i])
    ax.text(1.1, .44, "Optimal Precision: %.2f" % prec[best_i])
    ax.set_xlabel('Discriminative Threshold')
    return ax


def cv_to_df(cv):
    cv_results = pd.DataFrame(list(cv['params']))
    for col in ['split2_test_score', 'split1_test_score', 'split0_test_score', 'mean_test_score','std_test_score', 'mean_train_score']:
        cv_results[col] = cv[col]

    cv_results['sharpe'] = cv_results['mean_test_score'] / cv_results['std_test_score']
    cv_results['testing_over_training'] = cv_results['mean_test_score'] / cv_results['mean_train_score']    
    cv_results = cv_results.sort_values('sharpe')
    return cv_results

re_not_decimal = re.compile(r'[^\.0-9]*')

def process_int_rate(x):
    x = x.strip()
    x = re_not_decimal.sub("", x)    
    return float(x)

def process_revol_util(x):
    if pd.isnull(x):
        return 0
    else:
        x = x.strip()
        x = re_not_decimal.sub("", x)    
        return float(x)

def process_term(x):
    x = re_not_decimal.sub("", x)
    return int(x)

def process_emp_length(x):
    x = re_not_decimal.sub("", x)
    if x == '':
        return np.nan
    else:
        return float(x)


