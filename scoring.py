from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

from scipy.sparse import issparse
import matplotlib.pyplot as plt
from numpy import arange, matrix, unique, array_equal

def get_scores(target, prediction, avg_type):    
    return { 'accuracy': accuracy_score(target, prediction),
            'f1' : f1_score(target, prediction, average=avg_type),
            'recall' : recall_score(target, prediction, average=avg_type),
            'precision' : precision_score(target, prediction, average=avg_type) }

def graph_scores(scores, keys, title):
    count = len(scores)
    metric = tuple(t.title() for t in scores[0].keys())
    colors=['b', 'g', 'r', 'c', 'm', 'y']
    bar_width = .81 / count
    y_vals = arange(len(metric))
    
    for i, score in enumerate(scores):
        plt.bar(y_vals + bar_width * (i + ((3 - count) / 2)) , tuple(score.values()), width=bar_width, align='center', alpha=0.2, color=colors[i], linewidth=1, edgecolor='k')

    plt.legend(keys, bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.xticks(y_vals + bar_width, metric)
    plt.ylabel('Score')
    plt.ylim(0.,1.)
    plt.title(f'Evaluation Metrics for {title}\n\n')

    plt.show()
    
def print_scores(scores, avg_type):
    print(f'Accuracy: {scores["accuracy"]:0.4f}')
    print(f'{avg_type}-averaged F1: {scores["f1"]:0.4f}')
    print(f'{avg_type} averaged recall: {scores["recall"]:0.4f}')
    print(f'{avg_type} averaged precision: {scores["precision"]:0.4f}')

def graph_precision_recall(target, conf_list, titles, classes=[]):
    colors = iter(['b','r','g','c','m','k'])

    plt.figure()
    
    for confidence in conf_list:
        if issparse(target):
            target = target.toarray()
        
        # If target is multiclass it must be binarized
        if not array_equal(target, target.astype(bool)):
            classes_auto = unique(target)
            target = label_binarize(target, classes_auto if len(classes) < len(classes_auto) else classes)

        # A "micro-average": quantifying score on all classes jointly
        precision, recall, threshold = precision_recall_curve(target.ravel(), confidence.ravel())
        average_precision = average_precision_score(target, confidence, average="micro")

        
        plt.step(recall, precision, color=next(colors), alpha=.5,where='post')

    plt.legend(titles, bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curves')
    plt.show()