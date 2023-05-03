from sklearn.cluster import MiniBatchKMeans
import numpy as np

def kcluster(features):
    # fit on the whole data
    kmeans = MiniBatchKMeans(n_clusters=10,
                                random_state=0,
                                batch_size=10,
                                max_iter=10,
                                n_init = 'auto').fit(features)

    return kmeans



def select_diverse_samples(model, unlabeled_dataloader, unlabeled_dataset, device, n_clusters, n_samples_per_cluster, logger):
    features, paths = extract_features(model, unlabeled_dataloader, device, logger)
    # Perform clustering
    kmeans = kcluster(features)

    # Reduce dimensions using PCA or t-SNE
    reducer = PCA(n_components=2)  # Or use TSNE(n_components=2, random_state=0)
    reduced_embeddings = reducer.fit_transform(features)

    # Create a scatter plot with colored points based on their cluster
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=kmeans.labels_, cmap='viridis')

    # Set plot title and labels
    plt.title("Clustering of feature embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    # Save the plot to a file
    plt.savefig(os.path.join(cfg["saver"]["snapshot_dir"], "clustering_visualization.png"))
    plt.close()

    cluster_assignments = kmeans.predict(features)

    selected_samples = []
    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_assignments == cluster)[0]
        if len(cluster_indices) > n_samples_per_cluster:
            cluster_indices = np.random.choice(cluster_indices, n_samples_per_cluster, replace=False)
        selected_samples.extend(cluster_indices)

    selected_paths = [paths[i] for i in selected_samples]

    return selected_paths

