import os
import os.path as osp
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def kcluster(features):
    kmeans = MiniBatchKMeans(n_clusters=10,
                                random_state=0,
                                batch_size=10,
                                max_iter=10,
                                n_init = 'auto').fit(features)

    return kmeans

def select_diverse_samples(n_clusters, n_samples_per_cluster, n_lines=None):
    current_dir = osp.dirname(osp.abspath(__file__))
    file_dir = osp.join(current_dir, r"result_report_0.0_Unsupervised")
    features_dir = osp.join(file_dir, r"features.txt")
    paths_dir = osp.join(file_dir, r"paths.txt")
    
    with open(features_dir, 'r') as f:
        features = np.array([list(map(float, line.strip().split(','))) for line in f][:n_lines])

    with open(paths_dir, 'r') as f:
        paths = [line.strip() for line in f][:n_lines]

    kmeans = kcluster(features)

    reducer = PCA(n_components=2)
    reduced_embeddings = reducer.fit_transform(features)

    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=kmeans.labels_, cmap='viridis')

    plt.title("Clustering of feature embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    plt.savefig(os.path.join(file_dir, "clustering_visualization.png"))
    plt.close()

    cluster_assignments = kmeans.predict(features)

    selected_samples = []
    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_assignments == cluster)[0]
        if len(cluster_indices) > n_samples_per_cluster:
            cluster_indices = np.random.choice(cluster_indices, n_samples_per_cluster, replace=False)
        selected_samples.extend(cluster_indices)

    selected_paths = [paths[i] for i in selected_samples]

    # Save the selected paths to a file
    with open(os.path.join(file_dir, "annotation_queue.txt"), 'w') as f:
        for path in selected_paths:
            f.write(f"{path}\n")

    return selected_paths

selected_paths = select_diverse_samples(n_clusters=10, n_samples_per_cluster=34)

