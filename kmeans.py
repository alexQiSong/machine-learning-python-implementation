#############################################################
# By Qi Song
# 04/28/17
#############################################################

from random import randrange

# Check if two assignments are the same
def check_assignment(assignment1,assignment2):
    for i in range(len(assignment1)):
        if assignment1[i] != assignment2[i]:
            return False
    return True

'''
'data' is an array. 'k' is the number of centroids.
'max_iter' is the maximum number of iterations.
This function returns the label for each data point
'''
def Kmeans(data,k,max_iter=300):

    # Initialization, randomly choose k points as k centroids
    centroids_index = [randrange(len(data)-1) for i in range(k)]
    centroids = [data[i] for i in range(len(centroids_index))]

    # assignment for the previous step and the current step
    assignment_previous = [-1]*len(data)
    assignment_current = [-1]*len(data)

    # Iterative clustering process
    for i in range(max_iter):

        '''
        Iteration step one. Calculate the euclidean distance between each data point and
        each centroid. Assign each datapoint to the closest centroid. 'assignment' stores
        the centroid assignment for each data point
        '''
        for i in range(len(data)):
            min_eu = float('Inf')
            min_eu_centroid = -1

            for j in range(len(centroids)):
                # Calculate distance between data point and the centroid. distance = squared euclidean distance
                current_eu = sum([(data[i][k]-centroids[j][k])**2 for k in range(len(data[0]))])
                if current_eu < min_eu:
                    min_eu = current_eu
                    min_eu_centroid = j
            assignment_current[i] = min_eu_centroid

        '''
        Iteration step two. Update the centroids. New centroids were calculated using
        the average of all the data points in current cluster.
        '''
        for i in range(len(centroids)):
            centroids[i] = [0]*len(centroids[0])
            cluster_size = 0

            for j in range(len(assignment_current)):
                if i==assignment_current[j]:
                    cluster_size += 1
                    for k in range(len(centroids[i])):
                        centroids[i][k] += data[j][k]

            # Average value
            for k in range(len(centroids[i])):
                centroids[i][k] /= float(cluster_size)

        # If the assignment does not change, the loop will end. Otherwise continue the loop
        if check_assignment(assignment_current,assignment_previous):
            break
        else:
            assignment_previous = assignment_current

    return assignment_current