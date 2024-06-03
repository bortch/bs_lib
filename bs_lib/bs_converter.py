import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import bs_lib.bs_terminal as term


class TypeConverter(BaseEstimator, TransformerMixin):
    def __init__(self, type="float") -> None:
        self.type = type

    def fit(self, X, y=None):
        # préparer la transformation
        return self

    def transform(self, X):
        # retourner la transformation
        return self.type_convert(X)  # transformé

    def type_convert(self, X):
        return X.astype(eval(self.type))


class Discretizer(BaseEstimator, TransformerMixin):
    def __init__(self, target, kwargs) -> None:
        """Init with pandas cut arguments as kwargs

        Args:
            kwargs (dict): [description]. example {'bins': [0,1,2,3], 'labels': ["A","B","C"], 'retbins': False}.
        """
        self.kwargs = kwargs
        self.target = target

    def set_params(self, target, kwargs):
        self.kwargs = kwargs
        self.target = target

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = pd.DataFrame(X).copy()
        print(self.target)
        for t in self.target:
            key = f"{t}_category"
            X_[key] = pd.cut(X[t], **self.kwargs).cat.codes
        # https://stackoverflow.com/questions/56018238/python-float-argument-must-be-a-string-or-a-number-not-pandas-libs-interva
        print(X_.info())
        return X_


if __name__ == "__main__":
    test = pd.DataFrame(
        {
            "col1": [10, 20, 30, 9],
            "col2": [5, 0, 15, 67],
            "col3": [40, 10, 1, 20],
        }
    )
    term.article(
        title="Initial",
        content=term.create_dataframe(title="test dataframe", content=test),
    )

    discretizer = Discretizer(
        ["col1", "col2"],
        {"bins": [-1, 25, 75, 100], "labels": ["A", "B", "C"]},
    )

    test = discretizer.fit_transform(test)

    term.article(
        title="Discretized",
        content=term.create_dataframe(title="test dataframe", content=test),
    )
