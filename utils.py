from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN, MeanShift

algorithms = {
    # 'RandomForest': RandomForestClassifier(),
    # 'DecisionTree': DecisionTreeClassifier(),
    # 'KNN': KNeighborsClassifier()
    'KMeans': KMeans(n_clusters=3, max_iter=300),
    'DBSCAN': DBSCAN(),
    'MeanShift': MeanShift()
}
