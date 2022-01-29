import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, v_measure_score, \
    davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import utils


class Dashboard(object):
    def __init__(self):
        self.df = pd.read_csv("data/packetbeat_mix.csv")
        self.X = self.df  # self.df.iloc[:, 1:14]
        self.model = None
        self.y_pred = None
        self.df_norm = self.X  # pd.DataFrame(MinMaxScaler().fit_transform(self.X))
        self.df_norm.columns = self.X.columns
        self.wcss = []
        self.silhouette = []
        self.calinski = []
        self.davies = []
        self.pca = pd.DataFrame(PCA(n_components=2).fit_transform(self.df_norm))
        cols = ['Coord 1', 'Coord 2']
        self.pca.columns = cols
        self.df_no_outliers = None

        # wine_true = pd.DataFrame(self.df['Wine'])

        # self.true_labels = np.array([], int)
        # for i in wine_true:
        # self.true_labels = np.append(self.true_labels, wine_true[i] - 1)

        for i in range(2, 11):
            kmeans = KMeans(n_clusters=i, max_iter=300)
            pred = kmeans.fit_predict(self.df_norm)
            self.wcss.append(kmeans.inertia_)
            self.silhouette.append(silhouette_score(self.df_norm, pred))
            self.calinski.append(calinski_harabasz_score(self.df_norm, pred))
            self.davies.append(davies_bouldin_score(self.df_norm, pred))

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
        else:
            self.model = algorithm.fit(self.df_norm)
            self.y_pred = self.model.fit_predict(self.df_norm)
            self.pca['Labels'] = algorithm.labels_
            self.df_no_outliers = pd.DataFrame(self.pca)
            indexNames = self.df_no_outliers[self.df_no_outliers['Labels'] == -1].index
            self.df_no_outliers.drop(indexNames, inplace=True)

    def get_indicators(self):
        silhouette = silhouette_score(self.pca, self.y_pred)
        silhouette = round(silhouette, 4)
        homogeneity = 1  # homogeneity_score(self.true_labels, self.y_pred)
        homogeneity = round(homogeneity, 4)
        completeness = 1  # completeness_score(self.true_labels, self.y_pred)
        completeness = round(completeness, 4)
        vmeasure = 1  # v_measure_score(self.true_labels, self.y_pred)
        vmeasure = round(vmeasure, 4)
        calinski = calinski_harabasz_score(self.pca, self.y_pred)
        calinski = round(calinski, 4)
        davies = davies_bouldin_score(self.pca, self.y_pred)
        davies = round(davies, 4)
        return silhouette, homogeneity, completeness, vmeasure, calinski, davies

    def get_variable_names(self):
        variables = []
        for col in self.X.columns:
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
