import math
import pandas as pd
import numpy as np

from scipy.stats import iqr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import bs_lib.bs_file as bsf

tfont = {'fontsize': 15, 'fontweight': 'bold'}
sns.set_style('darkgrid')

# DEPRECATED
def get_list_dir(in_directory_path, with_extension=None, match_terms=None, exclude_terms=None, separator=".", verbose=False):
    print('bs_eda.get_list_dir deprecated use bs_file.get_list_dir')
    return bsf.get_list_dir(in_directory_path=in_directory_path, with_extension=with_extension, match_terms=match_terms, exclude_terms=exclude_terms, separator=separator, verbose=verbose)

#DEPRECATED
def load_all_csv(dataset_path="dataset", exclude=None, index=None, verbose=False):
    print('bs_eda.load_all_csv deprecated use bs_file.load_all_csv')
    return bsf.load_all_csv(dataset_path=dataset_path, exclude=exclude, index=index, verbose=verbose)


def get_numerical_columns(data):
    return list(data.select_dtypes(include=[np.number]).columns.values)


def get_categorical_columns(data):
    return list(data.select_dtypes(include=['object', 'category']).columns.values)


def get_histplot(data, column, log=False, rotation=0):
    plt.figure(figsize=(12, 7))
    column_name = column
    if log:
        column_name = f'log({column_name})'
    plt.title(f'\nHistplot for {column_name}', fontdict=tfont)
    if log == True:
        data = np.log(data[column])
    else:
        data = data[column]
    sns.histplot(data=data, fill=True, kde=True)
    plt.xticks(rotation=rotation)
    plt.show()


def get_countplot(data, column):
    plt.figure(figsize=(12, 7))
    plt.title(f'\n Countplot for {column}', fontdict=tfont)
    sns.countplot(x=data[column])
    plt.show()
    print(data[column].value_counts(normalize=True))


def get_ecdf(data, col):
    plt.figure(figsize=(12, 7))
    plt.title(f'ECDF for {col}', fontdict=tfont)
    sns.ecdfplot(data=data, x=col, hue='stroke')
    plt.axhline(0.5, c='red', linestyle='dashed')
    plt.show()


def get_kde(data, col, target, hue=None):
    target_values = data[target].unique()
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    i = -1
    hue_ = None
    if hue:
        hue_ = data[hue]
    for v in target_values:
        i += 1
        c = colors[i % len(colors)]
        plt.figure(figsize=(12, 7))
        plt.title(f"kde {col} vs {target}: {v}", fontdict=tfont)
        sns.kdeplot(x=data[col][data[target] == v], label=v,
                    fill=True, hue=hue_, legend=True)
        plt.axvline(data[col].mean(),
                    c='k',
                    linestyle='dashed',
                    label=f'mean {col}')
        plt.axvline(data[col][data[target] == v].mean(),
                    linestyle='dashed',
                    color=c,
                    label=f'mean {col} for {v}')
        plt.legend()
        plt.show()
        print(
            f"Avg {col} for {target} == {v}: {data[col][data[target]==v].mean()}\n")


def get_kde_continue(data, col, target):
    plt.figure(figsize=(12, 7))
    plt.title(f"Kernel Density Estimation for {col}", fontdict=tfont)
    sns.kdeplot(x=data[col], y=data[target], fill=True)
    plt.axvline(data[col].mean(), label=f'{col} mean',
                c='red', linestyle='dashed')
    plt.axhline(data[target].mean(), label=f'{target} mean',
                c='blue', linestyle='dashed')
    plt.legend()
    plt.show()
    print(f"Avg {col}: {data[col].mean()}")
    print(f"Avg {target}: {data[target].mean()}")


def get_relplot(data, x, y, hue=None, x_trace=['mean'], y_trace=['mean']):
    #plt.figure(figsize=(12, 7))
    if hue:
        hue_ = hue
    else:
        hue_ = y
    sns.relplot(x=x, y=y, data=data, hue=hue_, height=6, aspect=1.6)
    plt.title(f"Relation plot for {x} vs {y}", fontdict=tfont)
    for i in range(len(x_trace)):
        if x_trace[i] == 'mean':
            plt.axvline(data[x].mean(), label=f'{x} mean',
                        c='red', linestyle='dashed')
        if x_trace[i] == 'median':
            plt.axvline(data[x].median(), label=f'{x} median',
                        c='green', linestyle='dashed')
        if x_trace[i] == 'mode':
            plt.axvline(data[x].mode()[0], label=f'{x} mode',
                        c='cyan', linestyle='dashed')

    for i in range(len(y_trace)):
        if y_trace[i] == 'mean':
            plt.axhline(data[y].mean(), label=f'{y} mean',
                        c='blue', linestyle='dashed')
        if y_trace[i] == 'median':
            plt.axhline(data[y].median(), label=f'{y} median',
                        c='orange', linestyle='dashed')
        if y_trace[i] == 'mode':
            plt.axhline(data[y].mode()[0], label=f'{y} mode',
                        c='gray', linestyle='dashed')

    plt.legend()
    plt.show()
    print(f"Avg {x}: {data[x].mean()}")
    print(f"Avg {y}: {data[y].mean()}")


def get_lmplot(data, x, y, hue=None):
    #plt.figure(figsize=(12, 7))
    sns.lmplot(x=x, y=y, data=data,
               robust=True, palette='tab10', hue=hue,
               scatter_kws=dict(s=60, linewidths=.7, edgecolors='black'))
    plt.title(f"Relation plot for {x} vs {y}", fontdict=tfont)
    plt.axvline(data[x].mean(), label=f'{x} mean',
                c='red', linestyle='dashed')
    plt.axhline(data[y].mean(), label=f'{y} mean',
                c='blue', linestyle='dashed')
    plt.legend()
    plt.show()
    print(f"Avg {x}: {data[x].mean()}")
    print(f"Avg {y}: {data[y].mean()}")


def get_joinplot(data, col, target):
    g = sns.jointplot(x=data[col], y=data[target])
    g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
    g.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)
    plt.show()
    print(f"Avg {col}: {data[col].mean()}")
    print(f"Avg {target}: {data[target].mean()}")


def get_ecdf(data, col, target=None, mean=True, median=True):
    plt.figure(figsize=(12, 7))
    plt.title(f'ECDF for {col}', fontdict=tfont)
    if target:
        sns.ecdfplot(data=data, x=col, hue=target)
    else:
        sns.ecdfplot(data=data, x=col)
    if mean:
        plt.axvline(data[col].mean(), c='blue',
                    label='mean', linestyle='dashed')
    if median:
        plt.axvline(data[col].median(), c='orange',
                    label='median', linestyle='dashed')
    plt.axhline(0.5, c='red', linestyle='dashed')
    plt.legend()
    plt.show()


def get_boxplot(data, x, y=None, orient='v', rotation=0):
    plt.figure(figsize=(12, 7))
    if y:
        plt.title(f"Boxplot for {x} vs {y}", fontdict=tfont)
        sns.boxplot(x=data[x], y=data[y], orient=orient)
        plt.axhline(data[y].mean(), label=f'{y} mean',
                    c='blue', linestyle='dashed')
        plt.xticks(rotation=rotation)
        plt.show()
        print(f"Avg {y}: {data[y].mean()}")
    else:
        plt.title(f"Boxplot for {x}", fontdict=tfont)
        sns.boxplot(x=data[x], orient=orient)
        plt.xticks(rotation=rotation)
        plt.show()


def get_pairplot(data, log=None, hue=None):
    data_ = data.copy()
    if isinstance(log,list) and (len(log) > 0):
        for i in range(len(log)):
            column = log[i]
            try:
                data_[column] = np.log(data[column])
            except Exception as e:
                print(column, e)

    sns.pairplot(data=data_,
                 corner=True,
                 hue=hue,
                 plot_kws={'line_kws': {'color': 'orange', 'linewidth': 1},
                           'scatter_kws': {'alpha': 0.6, 's': 20, 'edgecolor': 'w'}},
                 diag_kind='kde',
                 kind="reg")

    plt.show()


def get_std_evolution(data, target, start=0, stop=100, num=10):
    return get_evolution(data=data, target=target, value='std', start=start, stop=stop, num=num)


def get_evolution(data, target, value='std', start=0, stop=100, num=10):
    # plan evolution range
    range_space = np.linspace(start=start, stop=stop, num=num)
    # create a datafram
    df_columns = get_numerical_columns(data)
    results = pd.DataFrame(columns=df_columns)
    # for each step in the range_space
    j = start
    for i in range_space:
        if value == 'std':
            std_series = data[(j <= data[target]) & (
                data[target] < i)][df_columns].std()
        if value == 'mean':
            std_series = data[(j <= data[target]) & (
                data[target] < i)][df_columns].mean()
        if value == 'count':
            std_series = data[(j <= data[target]) & (
                data[target] < i)][df_columns].count()
        j = i
        # add the pivot
        results.loc[i] = std_series.array
    return results


def get_nrows_ncols(dim, verbose=False):
    """Get the number of rows and columns given a number of items

    Args:
        dim (int): the number of items to holds in rows and columns

    Returns:
        int,int: number of rows, number of columns
    """
    ncols = round(math.sqrt(dim))
    nrows = round(math.ceil(dim / math.sqrt(dim)))
    if verbose:
        print(f'dim:{dim}, n_cols:{ncols}, n_rows:{nrows}')
    return nrows, ncols


def get_features_as_grid_of_lineplot(data, dim=None, log=False):
    features = get_numerical_columns(data)
    if not dim:
        dim = len(features)
    nrows, ncols = get_nrows_ncols(dim)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))

    for i, subplot in zip(range(dim), ax.flatten()):
        if log:
            y = np.log(data[features[i]])
        else:
            y = data[features[i]]
        sns.lineplot(x=data.index, y=y, ax=subplot)
    plt.show()


def get_crosstab(data, x, y):
    temp = pd.crosstab(data[x], data[y],
                       normalize='columns')
    # afficher sous forme de frequence: normalize='columns'
    plt.figure(figsize=(12, 7))
    plt.title(f'Crosstab Heatmap for {x} vs {y} frequency\n', fontdict=tfont)
    sns.heatmap(temp, annot=True, cmap='Blues', cbar=False)
    plt.show()


def get_correlation(data, target=None, fields=None):
    if fields:
        _data = data.filter(items=fields)
    else:
        _data = data
    corr = _data.corr()
    if target:
        mask = None
        plt.figure(figsize=(2, 7))
        plt.title(f'Coefficient de Correlation: {target}\n', tfont)
        corr = corr.filter([target]).drop([target])
    else:
        plt.figure(figsize=(12, 7))
        plt.title('Correlation Coefficient \n', tfont)
        mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, cmap='vlag', cbar=False,
                fmt='.2f', square=True, mask=mask)
    plt.show()


def get_percent(part, total):
    return round((part/total)*100, 2)


def get_repartition(data, feature, threshold):
    total = data[feature].count()
    under_thresh = data[data[feature] < threshold][feature].count()
    over_thresh = data[data[feature] >= threshold][feature].count()
    pct_under = get_percent(under_thresh, total)
    pct_over = get_percent(over_thresh, total)
    return pct_under, pct_over


def get_occurence(data, feature, value, operator='=', show=False):
    if operator == '=':
        entries = data[data[feature] == value][feature]
    if operator == '>':
        entries = data[data[feature] > value][feature]
    if operator == '<':
        entries = data[data[feature] < value][feature]
    if operator == '>=':
        entries = data[data[feature] >= value][feature]
    if operator == '<=':
        entries = data[data[feature] <= value][feature]

    if show:
        n_entries = entries.count()
        if n_entries > 1:
            s = 'ies'
        else:
            s = 'y'
        print(
            f"There are {n_entries} entr{s} with a value {operator} {value} for the variable '{feature}'")
    else:
        return entries


def get_suspected_outliers(data, feature, show=False):
    # An observation is considered a suspected outlier if it is:
    # below Q1 - 1.5(IQR) or
    # above Q3 + 1.5(IQR)
    # The following picture illustrates this rule
    q1 = data[feature].quantile(.25)
    q3 = data[feature].quantile(.75)
    irq = q3-q1
    low_limit = q1 - 1.5*(irq)
    high_limit = q3 + 1.5*(irq)
    if show:
        print(f"Q1: {q1}\nQ3: {q3}\nIQR: {irq}")
        print(
            f"\nLower limit below which a value is suspect: {low_limit}")
        get_occurence(data=data, feature=feature,
                      value=low_limit, operator='<', show=True)
        print(
            f"\nThe upper limit beyond which a value is suspect: {high_limit}")
        get_occurence(data=data, feature=feature,
                      value=high_limit, operator='>', show=True)
    else:
        low_outlier = get_occurence(
            data=data, feature=feature, value=low_limit, operator='<')
        high_outlier = get_occurence(
            data=data, feature=feature, value=high_limit, operator='>')
        return low_outlier, high_outlier


def split_by_row(data, percentage):
    da = data.sample(frac=percentage)
    ta = data[~data.index.isin(da.index)]
    da = da.reset_index(drop=True)
    ta = ta.reset_index(drop=True)
    # print(da.info(),ta.info())
    return da, ta


def train_val_test_split(X, y, test_size, train_size, val_size, random_state=None, show=False):
    if isinstance(X, pd.DataFrame):
        X = X.reset_index(drop=True)
    if isinstance(y, pd.Series):
        y = y.reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=test_size/(test_size + val_size), random_state=random_state, shuffle=False)
    if show:
        print("\nSplitting into train, val and test sets")
        print(f"\tX_train: {X_train.shape}\n\tX_val: {X_val.shape}\n\tX_test: {X_test.shape}\n\ty_train: {y_train.shape}\n\ty_val: {y_val.shape}\n\ty_test: {y_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_similar_row(to_this_row, in_data, based_on_cols, show=False):
    subset = []  # collecting used feature to filter
    index = 0  # feature index
    # temporary Dataframe that will be filtered out @ each iteration
    df_temp = in_data.copy()
    # pd.DataFrame(columns=in_data.columns.to_list()) # the result DataFrame
    df_result = in_data.copy()
    df_iteration_result = in_data.copy()
    min_not_found = True  # cdt to start/end the loop
    while min_not_found:
        feature = based_on_cols[index]
        df_iteration_result = in_data.where(
            cond=(df_temp[feature] == to_this_row[feature]), other=df_iteration_result)

        if not df_iteration_result.empty:
            subset.append(feature)
            df_temp = df_iteration_result
            df_result = df_iteration_result

        if index < len(based_on_cols)-1:
            index += 1
        else:
            min_not_found = False
            if show:
                print(
                    f"\nSolution:\nrow:\n{to_this_row}\nfeature:\n{subset}\nresult:\n{df_result.describe()}\n")
            return df_result
    return df_result


def unbiased_cramer_v(x, y, show=False):
    # It ranges from 0 to 1 where:
    #  * 0 indicates no association between the two variables.
    #  * 1 indicates a strong association between the two variables.
    crosstab = pd.crosstab(x, y)
    chi2, pvalue, _, _ = chi2_contingency(crosstab.values)
    n = crosstab.values.sum()
    r, k = crosstab.shape
    phi2d = max(0, chi2/n-((k-1)*(r-1))/(n-1))
    kd = k-(k-1)**2/(n-1)
    rd = r-(r-1)**2/(n-1)
    cramer_v = np.sqrt(phi2d/min(kd-1, rd-1))
    if show:
        print(
            f"\nchi2: {chi2:.4f}\npvalue: {pvalue:.4f}\nCramer V: {cramer_v:.4f}\n")
    return chi2, pvalue, cramer_v


def get_ordered_categories(data, by):
    df = data.copy()
    categories = {}
    columns = get_categorical_columns(df)
    for cat in columns:
        ordered_df = df[[cat, by]]
        ordered_df = ordered_df.groupby(cat).agg('mean').reset_index()
        ordered_df.sort_values(
            by, ascending=True, inplace=True, ignore_index=True)
        categories[cat] = []
        for c in ordered_df[cat].values:
            categories[cat].append(c)
    return categories
