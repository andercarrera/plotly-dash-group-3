from sklearn.cluster import KMeans, DBSCAN, MeanShift

algorithms = {
    'KMeans': KMeans(n_clusters=3, max_iter=300),
    'DBSCAN': DBSCAN()
}
