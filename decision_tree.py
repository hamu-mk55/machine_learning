import glob
import csv
import pickle

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,\
    ConfusionMatrixDisplay,\
    RocCurveDisplay,\
    PrecisionRecallDisplay

import matplotlib.pyplot as plt
import pandas as pd


class DecisionTreeModel:
    def __init__(self):
        self.model: DecisionTreeClassifier = None

        self.info_cols: list[int] | None = None
        self.x_cols: list[int] | None = None
        self.y_col: int | None = None

        self.x_names = None
        self.info_names = None
        self.y_name = None
        self.y_vals = None

    def set_params(self,
                   x_cols: list[int],
                   y_col: int,
                   info_cols: list[int] | None = None,
                   debug: bool = False):
        self.x_cols = list(set(x_cols))
        self.y_col = y_col
        self.info_cols = list(set(info_cols))

        if debug:
            print("x:\t", self.x_cols)
            print("y:\t", self.y_col)
            print("info:\t", self.info_cols)

    def train(self, df_train: pd.DataFrame, max_depth=5):

        df = self._check_dataframe(df_train, mode="train")

        X = self._make_x(df)
        Y = self._make_y(df)

        self.model = DecisionTreeClassifier(max_depth=max_depth)
        self.model = self.model.fit(X, Y)

        print("score: ", self.model.score(X, Y))
        print(self.model.feature_importances_)


        plt.figure(figsize=(80, 60))
        plot_tree(self.model,
                  feature_names=self.x_names,
                  class_names=[str(val) for val in self.y_vals],
                  filled=True)
        plt.savefig("plot_tree.png")
        plt.clf()

        importances = pd.DataFrame({'Importance': self.model.feature_importances_}, index=self.x_names)
        importances.sort_values('Importance', ascending=False, inplace=True)
        importances.to_csv("feature_importance.csv")

    def predict(self, df_test: pd.DataFrame):
        df = self._check_dataframe(df_test, mode="test")

        X = self._make_x(df)

        Y_PRED = self.model.predict(X)
        df = df.assign(PRED=Y_PRED)

        df.to_csv("results.csv")

        # 結果出力
        _df = self._delete_illegals_row(df, delete_nan_in_y=True)
        _Y = self._make_y(_df)
        _Y_PRED = _df["PRED"]

        print(confusion_matrix(_Y, _Y_PRED))
        print(classification_report(_Y, _Y_PRED))


        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), constrained_layout=True)
        ConfusionMatrixDisplay.from_predictions(_Y, _Y_PRED, ax=axes)
        plt.savefig("confusion.png")
        plt.clf()

    def _check_dataframe(self, df: pd.DataFrame, mode="test"):
        header = df.columns.tolist()
        self.x_names = [header[col_no] for col_no in self.x_cols]
        self.info_names = [header[col_no] for col_no in self.info_cols]
        self.y_name = header[self.y_col]
        self.y_vals = df[self.y_name].unique()

        if mode=="train":
            delete_nan_in_y = True
        else:
            delete_nan_in_y = False

        df = self._delete_illegals_row(df, delete_nan_in_y=delete_nan_in_y)

        return df

    def _delete_illegals_row(self,
                             df: pd.DataFrame,
                             delete_nan_in_y: bool = False):
        # delete if x in NaN
        df = df.dropna(subset=self.x_names)

        # delete if y in NaN
        if delete_nan_in_y:
            df = df.dropna(subset=[self.y_name])

        return df

    def _make_x(self, df: pd.DataFrame):
        X = df.loc[:, self.x_names]

        return X

    def _make_y(self, df: pd.DataFrame):

        Y = df.loc[:, self.y_name]

        return Y

    def _make_info(self):
        pass


def main():
    df = pd.read_csv("test.csv")

    model = DecisionTreeModel()

    _x_cols = [6, 2] + list(range(1, 8)) + [5, 4]
    _y_col = 9
    _info_cols = [0]
    model.set_params(_x_cols, _y_col, _info_cols, debug=False)

    model.train(df, max_depth=3)

    model.predict(df)

    return





if __name__ == '__main__':
    main()
