import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np

import utils


class Dashboard(object):
    def __init__(self):
        self.df = pd.read_csv("data/wine.csv")
        self.X = self.df.iloc[:, 1:14]
        y = self.df.iloc[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(self.X, y, test_size=0.20, random_state=0, stratify=y)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.y_pred = None
        self.explainer = None
        self.df_norm = pd.DataFrame(MinMaxScaler().fit_transform(self.X))
        self.df_norm.columns = self.X.columns
        self.wcss = []
        self.pca = pd.DataFrame(PCA(n_components=2).fit_transform(self.df_norm))
        cols = ['Coord 1', 'Coord 2']
        self.pca.columns = cols
        self.df_no_outliers = None

        wine_true = pd.DataFrame(self.df['Wine'])

        self.true_labels = np.array([], int)
        for i in wine_true:
            # r[i] = e[i]-1
            self.true_labels = np.append(self.true_labels, wine_true[i] - 1)


        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, max_iter=300)
            kmeans.fit(self.df_norm)
            self.wcss.append(kmeans.inertia_)

        kmeans = KMeans(n_clusters=3, max_iter=300)
        kmeans.fit(self.df_norm)
        self.y_pred = kmeans.predict(self.df_norm)
        self.pca['Labels'] = kmeans.labels_

    def update_model(self, algorithm_name):
        algorithm = utils.algorithms[algorithm_name]
        if algorithm_name == 'KMeans':
            self.model = algorithm.fit(self.df_norm)
            self.y_pred = self.model.predict(self.df_norm)
            self.pca['Labels'] = algorithm.labels_
        elif algorithm_name == 'DBSCAN':
            self.model = algorithm.fit(self.df_norm)
            self.y_pred = self.model.fit_predict(self.df_norm)
            self.pca['Labels'] = algorithm.labels_
            self.df_no_outliers = pd.DataFrame(self.pca)
            indexNames = self.df_no_outliers[self.df_no_outliers['Labels'] == -1].index
            self.df_no_outliers.drop(indexNames, inplace=True)
        else:
            self.model = algorithm.fit(self.X)
            self.y_pred = self.model.fit_predict(self.X)
            self.pca['Labels'] = algorithm.labels_
        #else:
         #   self.model = algorithm.fit(self.X_train, self.y_train)
          #  self.y_pred = self.model.predict(self.X_test)

    def get_indicators(self):
        silhouette = silhouette_score(self.pca, self.y_pred)
        homogeneity = homogeneity_score(self.true_labels, self.y_pred)
        completeness = completeness_score(self.true_labels, self.y_pred)
        vmeasure = v_measure_score(self.true_labels, self.y_pred)
        return silhouette, homogeneity, completeness, vmeasure

    def get_instances(self):
        options = []
        value = 0
        for i, instance in enumerate(self.y_test):
            option = {
                'label': "Instance " + str(i) + " (Real=" + str(instance) + " Pred=" + str(self.y_pred[i]) + ")",
                'value': str(i)
            }
            options.append(option)
        return options, value

    def get_shap_values(self, instance_number):
        if instance_number is None:
            return dict()
        i_shap = self.explainer.shap_values(self.X_test.iloc[int(instance_number)])
        if len(i_shap) == 2:
            i_shap = i_shap[1]
        else:
            i_shap = i_shap[0]
        dic = {}
        for i, shap in enumerate(i_shap):
            dic[self.df.columns[i]] = shap
        return dic

    def get_variable_names(self):
        variables = []
        for col in self.X_test.columns:
            var = {
                'label': col,
                'value': col
            }
            variables.append(var)
        return variables

    def get_columns(self, column_names):
        columns = self.df[column_names]
        return columns

    def update_k_param(self, value):
        algorithm = utils.algorithms['KMeans']
        algorithm.n_clusters = value
        self.model = algorithm.fit(self.df_norm)
        self.y_pred = self.model.predict(self.df_norm)
        self.pca['Labels'] = algorithm.labels_

    def update_dbscan_params(self, eps, min_samples):
        algorithm = utils.algorithms['DBSCAN']
        algorithm.eps = eps
        algorithm.min_samples = min_samples
        self.model = algorithm.fit(self.df_norm)
        self.pca['Labels'] = algorithm.labels_
        self.df_no_outliers = pd.DataFrame(self.pca)
        indexNames = self.df_no_outliers[self.df_no_outliers['Labels'] == -1].index
        self.df_no_outliers.drop(indexNames, inplace=True)
