import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn import metrics

# ,train_test_split, GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

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


def show_elbow(data, max_iter=10,title=''):
    nb_clusters = range(max_iter)
    inertia = np.empty(max_iter)
    for i in nb_clusters:
        km = KMeans(n_clusters=i+1, random_state=1).fit(data)
        inertia[i] = km.inertia_
    plt.figure(figsize=(10, 6))
    plt.title(f'Nbr of Clusters - Elbow Method {title}', fontsize=14)
    plt.plot(nb_clusters, inertia)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Total Inertia")
    #plt.legend()
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


def get_silhouette(X, max_cluster=12, min_cluster=2, show=False):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    km_silhouette = []
    if min_cluster < 2:
        min_cluster = 2
    for i in range(min_cluster, max_cluster+1):
        km = KMeans(n_clusters=i, random_state=0).fit(X_scaled)
        preds = km.predict(X_scaled)
        silhouette = silhouette_score(X_scaled, preds)
        km_silhouette.append(silhouette)
        if show:
            print(f"Silhouette score for {i} cluster(s): {silhouette:.4f}")

    plt.figure(figsize=(7, 4))
    plt.title(
        "The silhouette coefficient method \nfor determining number of clusters\n", fontsize=16)
    plt.scatter(x=[i for i in range(min_cluster, max_cluster+1)],
                y=km_silhouette, s=150, edgecolor='k')
    plt.grid(True)
    plt.xlabel("Number of clusters", fontsize=14)
    plt.ylabel("Silhouette score", fontsize=15)
    plt.xticks([i for i in range(min_cluster, max_cluster+1)], fontsize=14)
    plt.yticks(fontsize=15)
    plt.show()


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), show=True,savefig=False):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    if savefig:
        plt.savefig(f'Learning_curve_{title}.png')
    if show:
        plt.show()

def get_learning_curve(model, X, y, scoring, show=True, savefig=False):
    # scoring value: 'neg_mean_squared_error', 'recall', ...
    n, train_score, val_score = learning_curve(
        model, X, y, cv=5, train_sizes=np.linspace(0.3, 1, 5), scoring=scoring, n_jobs=-1)
    plt.figure(figsize=(12, 7))
    plt.title('Learning curve for %s' % model[len(model)-1], fontdict=tfont)
    plt.plot(n, train_score.mean(axis=1), label='Train score')
    plt.plot(n, val_score.mean(axis=1), label='Validation score')
    plt.xlabel('n')
    plt.ylabel(scoring)
    plt.legend()
    if savefig:
        plt.savefig(f'Learning_curve_{model[len(model)-1]}.png')
    if show:
        plt.show()

def plot_validation_curve(model,X,y,param_name,param_range):

    param_range = np.logspace(-6, -1, 5)
    train_scores, test_scores = validation_curve(
        model, X, y, param_name="gamma", param_range=param_range,
        scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.title("Validation Curve with SVM")
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)  
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()