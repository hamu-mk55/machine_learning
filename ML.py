import glob
import csv
import pickle
import time

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import pandas as pd


class MLModel:
    def __init__(self):
        self.model = None

        self.info_cols: list[int] | None = None
        self.x_cols: list[int] | None = None
        self.y_col: int | None = None

        self.x_names = None
        self.info_names = None
        self.y_name = None

    def set_params(self,
                   x_cols: list[int],
                   y_col: int,
                   info_cols: list[int] | None = None,
                   debug: bool = False):
        """
        set parameters
        :param x_cols: column-no for x
        :param y_col: column-no for y
        :param info_cols: column-no for infos (not use for x or y)
        :param debug:
        :return:
        """
        self.x_cols = list(set(x_cols))
        self.y_col = y_col
        self.info_cols = list(set(info_cols))

        if debug:
            print("x:\t", self.x_cols)
            print("y:\t", self.y_col)
            print("info:\t", self.info_cols)

    def set_model(self):
        assert NotImplementedError("set_model-function is not defined yet")

    def save_model(self, model_path="model.pickle"):
        with open(model_path, mode='wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, model_path="model.pickle"):
        with open(model_path, mode='rb') as f:
            self.model = pickle.load(f)

    def train(self, df_train: pd.DataFrame):
        df = self._check_dataframe(df_train, mode="train")

        X = self._make_x(df)
        Y = self._make_y(df)

        self.model.fit(X, Y)
        print(f"score: {self.model.score(X, Y): .4f}")

    def validate(self, df_valid: pd.DataFrame,
                 plot_file="confusion.png", save_plot: bool = False):
        df = self._check_dataframe(df_valid, mode="train")

        X = self._make_x(df)
        Y = self._make_y(df)
        Y_PRED = self.model.predict(X)
        df = df.assign(PRED=Y_PRED)

        print("\nconfusion_matrix")
        print(confusion_matrix(Y, Y_PRED))

        print("\nclassification_report")
        print(classification_report(Y, Y_PRED))

        if save_plot:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), constrained_layout=True)
            ConfusionMatrixDisplay.from_predictions(Y, Y_PRED, ax=axes)
            plt.savefig(plot_file)
            plt.clf()

    def predict(self, df_test: pd.DataFrame, csv_file="results.csv", save_csv: bool = False):
        df = self._check_dataframe(df_test, mode="test")

        X = self._make_x(df)
        Y_PRED = self.model.predict(X)
        df = df.assign(PRED=Y_PRED)

        if save_csv:
            df.to_csv(csv_file)

        return df

    def save_feature_importance(self, csv_file="feature_importance.csv"):
        if not hasattr(self.model, "feature_importances_"):
            return

        print(f'feature_importances: {self.model.feature_importances_}')

        importances = pd.DataFrame({'Importance': self.model.feature_importances_}, index=self.x_names)
        importances.sort_values('Importance', ascending=False, inplace=True)
        importances.to_csv(csv_file)

    def _check_dataframe(self, df: pd.DataFrame, mode="test"):
        header = df.columns.tolist()
        self.x_names = [header[col_no] for col_no in self.x_cols]
        self.info_names = [header[col_no] for col_no in self.info_cols]
        self.y_name = header[self.y_col]

        if mode == "train":
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
        x = df.loc[:, self.x_names]
        return x

    def _make_y(self, df: pd.DataFrame):
        y = df.loc[:, self.y_name]
        return y

    def _make_info(self, df: pd.DataFrame):
        info = df.loc[:, self.info_names]
        return info


class DecisionTreeModel(MLModel):
    def __init__(self):
        super().__init__()
        self.model: DecisionTreeClassifier | None = None

    def set_model(self, max_depth=5):
        self.model = DecisionTreeClassifier(max_depth=max_depth)

    def train(self, df_train: pd.DataFrame):
        super().train(df_train)

        self.validate(df_train, save_plot=False)
        self.save_feature_importance()

        self._save_plot_tree()

    def _save_plot_tree(self, y_vals: list | None = None, plot_file: str = "plot_tree.png"):
        plt.figure(figsize=(80, 60))

        plot_tree(self.model,
                  feature_names=self.x_names,
                  class_names=[str(val) for val in y_vals] if y_vals is not None else None,
                  filled=True)

        plt.savefig(plot_file)
        plt.clf()


class RandomForestModel(MLModel):
    def __init__(self):
        super().__init__()
        self.model: RandomForestClassifier | None = None

    def set_model(self):
        self.model = RandomForestClassifier(n_estimators=10)

    def train(self, df_train: pd.DataFrame):
        super().train(df_train)

        self.validate(df_train, save_plot=False)
        self.save_feature_importance()


class SVMModel(MLModel):
    def __init__(self):
        super().__init__()
        self.model: svm.SVC | None = None

    def set_model(self, kernel='rbf'):
        if kernel != "linear":
            self.model = svm.SVC(C=1,
                                 kernel=kernel,
                                 class_weight="balanced")
        else:
            self.model = svm.LinearSVC(C=1,
                                       class_weight="balanced")

    def train(self, df_train: pd.DataFrame):
        super().train(df_train)

        self.validate(df_train, save_plot=False)
        self.save_feature_importance()


def main():
    df = pd.read_csv("test.csv")

    _x_cols = [6, 2] + list(range(1, 8)) + [5, 4]
    _y_col = 9
    _info_cols = [0]

    models = {"DecisionTree": DecisionTreeModel(),
              "SVM": SVMModel(),
              "RandomForest": RandomForestModel()}

    for model_name, model in models.items():
        print(f'\n{model_name}........................................')

        model.set_params(_x_cols, _y_col, _info_cols, debug=False)
        model.set_model()

        model.train(df)

        time0 = time.time()
        model.predict(df.head(n=1))
        print(f"time[msec]: {(time.time() - time0) * 1000: .3f}")


if __name__ == '__main__':
    main()
