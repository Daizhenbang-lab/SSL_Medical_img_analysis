import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from sklearn.cluster import DBSCAN
'''

'''
def visualize_clusters_separately_from_csv(csv_path, eps=0.5, min_samples=1250):

    data = pd.read_csv(csv_path)

    umap = data[["UMAP1", "UMAP2", "UMAP3"]].values
    colors = data["color"].tolist()
    colors = [to_hex(color) for color in colors]


    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(umap)
    labels = clustering.labels_


    unique_labels = np.unique(labels[labels != -1])

    data['cluster_label'] = labels
    cluster_0_data = data[data['cluster_label'] == 0]
    cluster_0_data.to_csv('cluster_0_data.csv', index=False)  # save the cluster which we need



    for cluster_label in unique_labels:

        mask = labels == cluster_label
        umap_cluster = umap[mask]
        colors_cluster = np.array(colors)[mask]


        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')


        ax.scatter(umap_cluster[:, 0],
                   umap_cluster[:, 1],
                   umap_cluster[:, 2],
                   c=colors_cluster,
                   s=10)


        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_zlabel('UMAP3')


        plt.title(f'UMAP Projection - Cluster {cluster_label}')
        ax.view_init(elev=10, azim=135)


        plt.show()


        input(f'Press Enter to continue to the next cluster (Cluster {cluster_label})...')

csv_file_path = 'generated_result/updated_scan167.csv'
visualize_clusters_separately_from_csv(csv_file_path, eps=0.5, min_samples=1250)
