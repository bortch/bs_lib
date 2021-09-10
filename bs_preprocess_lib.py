import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn import metrics

# ,train_test_split, GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans

import json
#from sklearn.base import TransformerMixin

#from sklearn.preprocessing import StandardScaler, OneHotEncoder
#from sklearn.impute import SimpleImputer
# from sklearn.pipeline import make_pipeline # on utilise la version de imblearn
#from sklearn.compose import make_column_transformer
#from sklearn.decomposition import PCA

#from sklearn.tree import DecisionTreeClassifier, plot_tree
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LinearRegression

#from imblearn.over_sampling import RandomOverSampler
#from imblearn.pipeline import make_pipeline

# model persistence
# import pickle #(bien pour petit objet)
#from joblib import dump, load

tfont = {'fontsize': 15, 'fontweight': 'bold'}
sns.set_style('darkgrid')


def get_histplot(data, column, log=False):
    plt.figure(figsize=(12, 7))
    plt.title(f'\nHistplot for {column}', fontdict=tfont)
    if log == True:
        data = np.log(data[column])
    else:
        data = data[column]
    sns.histplot(data=data, fill=True, kde=True)
    plt.show()


def get_countplot(data, column):
    plt.figure(figsize=(12, 7))
    plt.title(f'\n Countplot for {column}+', fontdict=tfont)
    sns.countplot(x=data[column])
    plt.show()
    print(data[column].value_counts(normalize=True))


def get_ecdf(data, col):
    plt.figure(figsize=(12, 7))
    plt.title(f'ECDF for {col}', fontdict=tfont)
    sns.ecdfplot(data=data, x=col, hue='stroke')
    plt.axhline(0.5, c='red', linestyle='dashed')
    plt.show()


def get_kde(data, col, target):
    plt.figure(figsize=(12, 7))
    plt.title(f"kde {col}", fontdict=tfont)
    target_values = data[target].unique()
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    i = -1
    for v in target_values:
        i += 1
        c = colors[i % len(colors)]
        sns.kdeplot(x=data[col][data[target] == v], label=v, fill=True)
    # common_norm = False permet de ramener
    # les 2 kde sur une même échelle
    # on ajoute les moyennes conditionnelles
        plt.axvline(data[col].mean(),
                    c='k',
                    linestyle='dashed',
                    label=f'mean for {col}')
        plt.axvline(data[col][data[target] == v].mean(),
                    linestyle='dashed',
                    color=c,
                    label=f'mean for {v}')
        plt.legend()
        plt.show()
        print(
            f"Avg {col} for {target} == {v}: {data[col][data[target]==v].mean()}")


def get_kde_continue(data, col, target):
    plt.figure(figsize=(12, 7))
    plt.title(f"Kernel Density Estimation for {col}", fontdict=tfont)
    sns.kdeplot(x=data[col], y=data[target], fill=True)
    # common_norm = False permet de ramener
    # les 2 kde sur une même échelle
    # on ajoute les moyennes conditionnelles
    plt.axvline(data[col].mean(), label=f'{col} mean',
                c='red', linestyle='dashed')
    plt.axhline(data[target].mean(), label=f'{target} mean',
                c='blue', linestyle='dashed')
    plt.legend()
    plt.show()
    print(f"Avg {col}: {data[col].mean()}")
    print(f"Avg {target}: {data[target].mean()}")


def get_ecdf(data, col, target):
    plt.figure(figsize=(12, 7))
    plt.title(f'ECDF for {col}', fontdict=tfont)
    sns.ecdfplot(data=data, x=col, hue=target)
    plt.axhline(0.5, c='red', linestyle='dashed')
    plt.show()


def get_crosstab(data, col, target):
    temp = pd.crosstab(data[col], data[target],
                       normalize='columns')
    # afficher sous forme de frequence: normalize='columns'
    plt.figure(figsize=(12, 7))
    plt.title(f'Crosstab Heatmap for {col} frequency\n', fontdict=tfont)
    sns.heatmap(temp, annot=True, cmap='Blues', cbar=False)
    plt.show()


def get_correlation(data):
    plt.figure(figsize=(12, 7))
    plt.title('Correlation Coefficient \n', tfont)
    sns.heatmap(data.corr(), annot=True, cmap='Blues', cbar=False)
    plt.show()

# def get_learning_curve(model, X_train, y_train):
#     n, train_score, val_score = learning_curve(model, X_train, y_train,
#                                                cv=5, scoring='recall',
#                                                train_sizes=np.linspace(0.3, 1, 5))
#     # train_score et val_score sont le résultat de 5 cross validation
#     # donc il faut prendre la moyenne
#     plt.figure(figsize=(12, 7))
#     plt.title('Learning Curve', fontdict=tfont)
#     plt.plot(n, train_score.mean(axis=1), label='train_score')
#     plt.plot(n, val_score.mean(axis=1), label='val_score')
#     plt.xlabel('n')
#     plt.ylabel('recall')
#     plt.legend()
#     plt.show()


def get_learning_curve(model, X, y, scoring):
    # scoring value: 'neg_mean_squared_error', 'recall', ...
    n, train_score, val_score = learning_curve(
        model, X, y, cv=5, train_sizes=np.linspace(0.3, 1, 4), scoring=scoring)
    plt.figure(figsize=(12, 7))
    plt.title('Learning curve for %s' % model[len(model)-1], fontdict=tfont)
    plt.plot(n, train_score.mean(axis=1), label='Train score')
    plt.plot(n, val_score.mean(axis=1), label='Validation score')
    plt.xlabel('n')
    plt.ylabel(scoring)
    plt.legend()
    plt.show()


def show_elbow(data, max_iter=10):
    nb_clusters = range(max_iter)
    inertia = np.empty(max_iter)
    for i in nb_clusters:
        km = KMeans(n_clusters=i+1, random_state=1).fit(data)
        inertia[i] = km.inertia_
    plt.figure(figsize=(12, 7))
    plt.plot(nb_clusters, inertia)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Total Inertia")
    plt.legend()
    plt.show()

# Learning curve
# imprimer la loss et l'accuracy
# suivant les epochs


def get_nn_learning_curve(history, title='', filename=None, show=False):
    '''history: tf History object with keys: ['loss', 'accuracy', 'val_loss', 'val_accuracy']
    '''
    get_loss_curve(loss=history['loss'],
                   val_loss=history['val_loss'],
                   title=title, filename=filename, show=show)
    get_accuracy_curve(
        accuracy=history['accuracy'],
        val_accuracy=history['val_accuracy'],
        title=title, filename=filename, show=show)


def get_loss_curve(loss, val_loss, title='', filename=None, show=False):
    plt.figure(figsize=(12, 7))
    # loss et val_loss

    plt.suptitle(f'Loss curve', fontsize=14)
    plt.title(title, fontsize=10)
    plt.plot(loss, label='train')
    plt.plot(val_loss, label='validation')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    if filename:
        plt.savefig(f'{filename}_loss_curve.png')
        if show == True:
            plt.show()
    else:
        plt.show()


def get_accuracy_curve(accuracy, val_accuracy, title='', filename=None, show=False):
    plt.figure(figsize=(12, 7))
    # accuracy et val_accuracy
    plt.suptitle(f'Accuracy curve', fontsize=14)
    plt.title(title, fontsize=10)
    plt.plot(accuracy, label='train')
    plt.plot(val_accuracy, label='validation')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    if filename:
        plt.savefig(f'{filename}_accuracy_curve.png')
        if show == True:
            plt.show()
    else:
        plt.show()


def get_reports(y, y_pred, classes=None, title='', filename=None, show=False):
    plt.figure(figsize=(12, 7))
    plt.suptitle(f'Confusion Matrix', fontsize=14)
    plt.title(title, fontsize=10)
    cm = confusion_matrix(y, y_pred)
    if classes:
        sns.heatmap(cm, annot=True, cmap='Blues',
                    cbar=False, fmt='d', xticklabels=classes, yticklabels=classes)
    else:
        sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True Label')
    plt.legend()
    if filename:
        plt.savefig(f'{filename}_confusion_matrix.png')
        if show == True:
            plt.show()
    else:
        plt.show()
    print(classification_report(y, y_pred))


def get_reports_using(model, X, y):
    # sur le validation set
    # on va prédite et comparer les prédictions du y_val
    # on peut voir où sont faites les erreurs de classification
    y_pred = model.predict(X)
    get_reports(y, y_pred)


def save_history(model_history, filename, raw_name=True):
    # Get the dictionary containing each metric and the loss for each epoch
    history_dict = model_history.history
    # Save it under the form of a json file
    if raw_name == True:
        filename = f'{filename}_history.json'
    json.dump(history_dict, open(filename, 'w'))


def load_history(filename, raw_name=True):
    if raw_name == True:
        filename = f'{filename}_history.json'
    return json.load(open(filename, 'r'))


def get_color(step=100):
    for x in np.linspace(0, 10, step):
        yield cm.rainbow(x)


def get_metrics_from_histories(histories):
    metrics = set()
    for key, history in histories.items():
        for k in history.keys():
            if not 'val_' in k:
                metrics.add(k)
    print(metrics)
    return metrics


def get_multi_curve(histories, metrics=None, title=None, filename=None, show=False):
    if not title:
        title = ' - '.join(list(histories.keys()))
    if not metrics:
        metrics = get_metrics_from_histories(histories)

    fig = plt.figure(tight_layout=True, figsize=(12, 10))
    spec = gridspec.GridSpec(nrows=len(histories),
                             ncols=len(metrics), figure=fig)
    metrics_string = ', '.join(metrics)
    fig.suptitle(f'{metrics_string} curve\nmodel: {title}', fontsize=14)

    color = get_color(round(100/len(histories)))

    for row, (key, history) in enumerate(histories.items()):
        col = 0
        c = next(color)
        for m in metrics:
            ax = fig.add_subplot(spec[row, col])
            key_metric = m
            key_metric_val = f'val_{m}'
            if m in history:
                #y_min = ((min(history[key_metric])+min(history[key_metric_val]))/2.)-0.1
                #y_max = ((max(history[key_metric])+max(history[key_metric_val]))/2.)+0.
                if m == 'accuracy':
                    y_ = round(max(history[key_metric_val]), 2)
                    x_ = np.argmax(history[key_metric_val])
                    note = f"max:{y_} @ epoch:{x_}"
                if m == 'loss':
                    y_ = round(min(history[key_metric_val]), 2)
                    x_ = np.argmin(history[key_metric_val])
                    note = f"min:{y_} @ epoch:{x_}"
                ax.set_title(f'{key} - {m}\n{note}')
                ax.plot(history[key_metric], label=f'train',
                        linestyle='dotted', color=c)
                ax.plot(history[key_metric_val], label=f'validation', color=c)
                ax.set_xlabel('epochs')
                ax.set_ylabel(m)
                ax.set_xlim(0)
            col += 1
    # save as file or show
    if filename:
        plt.savefig(f'{filename}_multi_curve.png')
        if show == True:
            plt.show()
    else:
        plt.show()


def multi_plot_best_results(data, params_values, compare=None, x_zoom=0, y_zoom=0,
                            labels=dict()):
    ''' plot, zoom and show best value for a param in some data
    data: data to plot (matrix)
    params_value: array of param's values tested (ndarray)
    compare: another matrix to plot (matrix with same shape of data)
    x_zoom: bandwith around the best value on x-axis (int|float)
    y_zoom: bandwith around the best value on y-axis (int|float)
    labels: labels to display (dict) {"data": "data", "compare": "comparison",
                                    "x_axis": "x", "y_axis": "y"} 
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _labels = {"data": "data",
               "compare": "comparison",
               "x_axis": "x",
               "y_axis": "y"}
    _labels.update(labels)
    plt.plot(params_values, data, label=_labels['data'])

    if type(compare) == 'numpy.ndarray':
        plt.plot(params_values, compare, label=_labels['compare'])

    ymax = data.max()
    xpos = data.argmax()
    x = range(len(data))
    xmax = x[xpos]

    ax.annotate(f'Best score:{round(ymax,3)}\nBest value:{xmax}',
                xy=(xmax, ymax), xytext=(xmax, ymax+0.05),
                arrowprops=dict(facecolor='black'),
                )
    ax_x_min = 0
    ax_x_max = len(data)
    ax_y_min = 0
    ax_y_max = ymax

    if(x_zoom > 0):
        ax_x_min = xpos-x_zoom
        ax_x_max = xpos+x_zoom

    ax.set_xlim(ax_x_min, ax_x_max)

    if(y_zoom > 0):
        ax_y_min = ymax-y_zoom
        ax_y_max = ymax+y_zoom

    ax.set_ylim(ax_y_min, ax_y_max)

    plt.ylabel(_labels['y_axis'])
    plt.xlabel(_labels['x_axis'])
    plt.legend()
    plt.show()
