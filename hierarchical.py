#############################################################
# By Qi Song
# 04/29/17
#############################################################

from numpy import array,where,min
from numpy import max,fill_diagonal,newaxis
from scipy.spatial.distance import cdist

'''
'data' is an 2-d numpy array. 'n_cluster' is the number of expected clusters.
This function implements the agglomerative algorithm for hierarchical clustering.
Clustering process will stop when the number of clusters reaches n_cluster.
Linkage method is complete linkage. This function returns the label for each
data point.
'''
def hierarchical(data,n_cluster):

    # Initialization, each data point is a cluster
    current_n_cluster = data.shape[0]
    cluster_IDs = cluster_assignment = array(range(data.shape[0]))

    # Pairwise euclidean distance
    dist_matrix = cdist(data,data)

    # Filled the diagonal with infinite. So the distance to itself is infinite
    fill_diagonal(dist_matrix,float('Inf'))

    # Cluster merging will keep going until it reaches the specified number of clusters
    while current_n_cluster > n_cluster:

        '''
        Step one. Merge the two closest clusters
        '''
        # find the minimum distance in the distance matrix. Get their cluster IDs
        # Search only applies on sub_dist_matrix, where the distances of current
        # active cluster IDs are stored
        sub_dist_matrix = dist_matrix[cluster_IDs[:,newaxis],cluster_IDs]
        min_index = where(sub_dist_matrix == min(sub_dist_matrix))
        cluster1_ID,cluster2_ID = cluster_IDs[[min_index[0][0],min_index[1][0]]]

        # Merge two closest clusters
        cluster_assignment[where(cluster_assignment==cluster2_ID)]=cluster1_ID

        # number of clusters decreased by one
        current_n_cluster -= 1

        # Update list of cluster IDs
        cluster_IDs = array(list(set(cluster_assignment)))

        '''
        Step two. Re-compute the distance between the merged cluster
        and all other clusters.
        '''
        merged_points = where(cluster_assignment==cluster1_ID)
        for i in range(len(cluster_IDs)):
            # Avoid calculating the distance with the cluster itself
            if cluster1_ID != cluster_IDs[i]:
                cluster_points = where(cluster_assignment==cluster_IDs[i])
                dist_matrix[cluster1_ID,cluster_IDs[i]] = max(cdist(data[merged_points],data[cluster_points]))

    # Make the cluster ID start from 0 and end with n_cluster-1
    old_to_new_ID = {}
    new_ID = 0
    for i in range(len(cluster_assignment)):
        if cluster_assignment[i] not in old_to_new_ID:
            old_to_new_ID[cluster_assignment[i]] = new_ID
            cluster_assignment[i] = new_ID
            new_ID += 1
        else:
            cluster_assignment[i] = old_to_new_ID[cluster_assignment[i]]

    return cluster_assignment
