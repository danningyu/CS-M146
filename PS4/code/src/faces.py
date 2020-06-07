"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Famous Faces
"""

# python libraries
import collections

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# libraries specific to project
import util
from util import *
from cluster import *
import time

######################################################################
# helper functions
######################################################################

def build_face_image_points(X, y) :
    """
    Translate images to (labeled) points.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), features (each row is one image)
        y     -- numpy array of shape (n,), targets
    
    Returns
    --------------------
        point -- list of Points, dataset (one point for each image)
    """
    
    n,d = X.shape
    
    images = collections.defaultdict(list) # key = class, val = list of images with this class
    for i in xrange(n) :
        images[y[i]].append(X[i,:])
    
    points = []
    for face in images :
        count = 0
        for im in images[face] :
            points.append(Point(str(face) + '_' + str(count), face, im))
            count += 1

    return points


def plot_clusters(clusters, title, average) :
    """
    Plot clusters along with average points of each cluster.

    Parameters
    --------------------
        clusters -- ClusterSet, clusters to plot
        title    -- string, plot title
        average  -- method of ClusterSet
                    determines how to calculate average of points in cluster
                    allowable: ClusterSet.centroids, ClusterSet.medoids
    """
    
    plt.figure()
    np.random.seed(20)
    label = 0
    colors = {}
    centroids = average(clusters)
    for c in centroids :
        coord = c.attrs
        plt.plot(coord[0],coord[1], 'ok', markersize=12)
    for cluster in clusters.members :
        label += 1
        colors[label] = np.random.rand(3,)
        for point in cluster.points :
            coord = point.attrs
            plt.plot(coord[0], coord[1], 'o', color=colors[label])
    plt.title(title)
    plt.show()


def generate_points_2d(N, seed=1234) :
    """
    Generate toy dataset of 3 clusters each with N points.
    
    Parameters
    --------------------
        N      -- int, number of points to generate per cluster
        seed   -- random seed
    
    Returns
    --------------------
        points -- list of Points, dataset
    """
    np.random.seed(seed)
    
    mu = [[0,0.5], [1,1], [2,0.5]]
    sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]
    
    label = 0
    points = []
    for m,s in zip(mu, sigma) :
        label += 1
        for i in xrange(N) :
            x = util.random_sample_2d(m, s)
            points.append(Point(str(label)+'_'+str(i), label, x))
    
    return points


######################################################################
# k-means and k-medoids
######################################################################

def random_init(points, k) :
    """
    Randomly select k unique elements from points to be initial cluster centers.
    
    Parameters
    --------------------
        points         -- list of Points, dataset
        k              -- int, number of clusters
    
    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part 2c: implement (hint: use np.random.choice)
    return np.random.choice(a=points, size=k, replace=False)
    ### ========== TODO : END ========== ###


def cheat_init(points) :
    """
    Initialize clusters by cheating!
    
    Details
    - Let k be number of unique labels in dataset.
    - Group points into k clusters based on label (i.e. class) information.
    - Return medoid of each cluster as initial centers.
    
    Parameters
    --------------------
        points         -- list of Points, dataset
    
    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part 2f: implement
    initial_points = []
    cluster_labels = {}
    for point in points:
        if point.label not in cluster_labels:
            cluster_labels[point.label] = []
        cluster_labels[point.label].append(point)
    
    for key in cluster_labels.keys():
        cluster = Cluster(cluster_labels[key])
        initial_points.append(cluster.medoid())
    
    return initial_points
    ### ========== TODO : END ========== ###


def kMeans(points, k, init='random', plot=False) :
    """
    Cluster points into k clusters using variations of k-means algorithm.
    
    Parameters
    --------------------
        points  -- list of Points, dataset
        k       -- int, number of clusters
        average -- method of ClusterSet
                   determines how to calculate average of points in cluster
                   allowable: ClusterSet.centroids, ClusterSet.medoids
        init    -- string, method of initialization
                   allowable: 
                       'cheat'  -- use cheat_init to initialize clusters
                       'random' -- use random_init to initialize clusters
        plot    -- bool, True to plot clusters with corresponding averages
                         for each iteration of algorithm
    
    Returns
    --------------------
        k_clusters -- ClusterSet, k clusters
    """
    
    ### ========== TODO : START ========== ###
    # part 2c: implement
    # Hints:
    #   (1) On each iteration, keep track of the new cluster assignments
    #       in a separate data structure. Then use these assignments to create
    #       a new ClusterSet object and update the centroids.
    #   (2) Repeat until the clustering no longer changes.
    #   (3) To plot, use plot_clusters(...).
    k_clusters = ClusterSet()
    k_clusters_old = k_clusters
    k_clusters_new = ClusterSet()
    cluster_points = {} # key: a centroid; value: points assoc. with that centroid
    i = 1
    if init == 'random':
        initial_pts = random_init(points, k)
    elif init == 'cheat':
        initial_pts = cheat_init(points)
    curr_centroids = initial_pts    
    while True:
        for point in points:
            min_dist = 100000
            best_centroid = None
            for centroid in curr_centroids:
                distance = point.distance(centroid)
                if distance<min_dist:
                    min_dist = distance
                    best_centroid = centroid
            if best_centroid not in cluster_points:
                cluster_points[best_centroid] = []
            cluster_points[best_centroid].append(point)
        
        for key in cluster_points.keys(): 
            k_clusters_new.add(Cluster(cluster_points[key]))
        cluster_points.clear()
        
        plot_title = "K means, iteration #%d" %i
        if init == 'cheat':
            plot_title += ', with cheat init'
        if plot:
            plot_clusters(k_clusters_new, title = plot_title, average=ClusterSet.centroids)
        # for curr_cent in k_clusters_new.centroids():
        #     print(curr_cent)
        if k_clusters_new.equivalent(k_clusters_old):
            break
        k_clusters_old = k_clusters_new
        k_clusters_new = ClusterSet()
        i = i+1
        curr_centroids = k_clusters_old.centroids()
        # print ''

    return k_clusters_old
    ### ========== TODO : END ========== ###


def kMedoids(points, k, init='random', plot=False) :
    """
    Cluster points in k clusters using k-medoids clustering.
    See kMeans(...).
    """
    ### ========== TODO : START ========== ###
    # part 2e: implement
    k_clusters = ClusterSet()
    k_clusters_old = k_clusters
    k_clusters_new = ClusterSet()
    cluster_points = {} # key: a medoid; value: points assoc. with that medoid
    i = 1
    if init == 'random':
        initial_pts = random_init(points, k)
    elif init == 'cheat':
        initial_pts = cheat_init(points)
    
    curr_medoids = initial_pts    
    while True:
        for point in points:
            min_dist = 100000
            best_medoid = None
            for medoid in curr_medoids:
                distance = point.distance(medoid)
                if distance<min_dist:
                    min_dist = distance
                    best_medoid = medoid
            if best_medoid not in cluster_points:
                cluster_points[best_medoid] = []
            cluster_points[best_medoid].append(point)
        
        for key in cluster_points.keys(): 
            k_clusters_new.add(Cluster(cluster_points[key]))
        cluster_points.clear()
        plot_title = "K medoids, iteration #%d" %i
        if init == 'cheat':
            plot_title += ', with cheat init'
        if plot:
            plot_clusters(k_clusters_new, title = plot_title, average=ClusterSet.medoids)
        # for curr_cent in k_clusters_new.medoids():
        #     print(curr_cent)
        if k_clusters_new.equivalent(k_clusters_old):
            break
        k_clusters_old = k_clusters_new
        k_clusters_new = ClusterSet()
        i = i+1
        curr_medoids = k_clusters_old.medoids()
        # print ''

    return k_clusters_old
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################

def main() :
    ### ========== TODO : START ========== ###
    # part 1: explore LFW data set
    X, y = get_lfw_data()
    # show_image(X[0])
    # show_image(np.mean(X, axis=0))
    U, mu = util.PCA(X)
    # plot_gallery([vec_to_image(U[:,i]) for i in xrange(12)])
    # l_values = [1, 10, 50, 100, 500, 1288]
    # for l_value in l_values:
    #     Z, UI = apply_PCA_from_Eig(X, U, l_value, mu)
    #     X_rec = reconstruct_from_PCA(Z, UI, mu)
    #     title_text = "Reconstructed for l = %d" %l_value
    #     plot_gallery(X_rec, title =title_text)

    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part 2d-2f: cluster toy dataset
    np.random.seed(1234)
    # print 'Problem 2(d)'
    # points_list = generate_points_2d(20)

    # only do one or the other, if you do both, the results appear to be wrong...
    # cluster_set_result = kMeans(points_list, 3, plot=True)
    # cluster_set_result2 = kMedoids(points_list, 3, plot=True)

    # using cheat_init
    # cluster_set_result3 = kMeans(points_list, 3, init='cheat', plot=True)
    # cluster_set_result4 = kMedoids(points_list, 3, init='cheat', plot=True)
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###    
    # part 3a: cluster faces
    np.random.seed(1234)
    X1, y1 = util.limit_pics(X, y, [4, 6, 13, 16], 40)
    points = build_face_image_points(X1, y1)
    k_means_purity = []
    k_medoids_purity = []
    k_means_times = []
    k_medoids_times = []

    for i in range(10):
        # repeat 10 times

        start_time = time.time()
        k_medoids_result = kMedoids(points, 4)
        end_time = time.time()
        k_medoids_purity.append(k_medoids_result.score())
        k_medoids_times.append(end_time - start_time)

        start_time = time.time()
        k_means_result = kMeans(points, 4)
        end_time = time.time()
        k_means_purity.append(k_means_result.score())
        k_means_times.append(end_time - start_time)      
    
    k_means_min = min(k_means_purity)
    k_means_max = max(k_means_purity)
    k_means_average = np.mean(np.asarray(k_means_purity))
    print 'K-means min: %f, max: %f, avg: %f' % (k_means_min, k_means_max, k_means_average)
    print 'K-means avg time: %f' % np.mean(np.asarray(k_means_times))

    print 'K-medoids min: %f, max: %f, avg: %f' % (min(k_medoids_purity), \
                                                    max(k_medoids_purity), \
                                                    np.mean(np.asarray(k_medoids_purity)))
    print 'K-medoids avg time: %f' % np.mean(np.asarray(k_medoids_times))


    # part 3b: explore effect of lower-dimensional representations on clustering performance
    # np.random.seed(1234)
    # X2, y2 = util.limit_pics(X, y, [4, 13], 40)
    # points = build_face_image_points(X2, y2)
    # component_list = np.arange(1, 42, 2)
    # k_means_results = []
    # k_medoids_results = []
    # for l_val in component_list:
    #     print(str(l_val))
    #     Z2, U2 = apply_PCA_from_Eig(X2, U, l_val, mu)
    #     X2new = reconstruct_from_PCA(Z2, U2, mu)
    #     points = build_face_image_points(X2new, y2)
    #     k_means_result = kMeans(points, 2, init='cheat')
    #     k_medoids_result = kMedoids(points, 2, init='cheat')
    #     k_means_results.append(k_means_result.score())
    #     k_medoids_results.append(k_medoids_result.score())

    # plt.title('Clustering Performance vs. # of Retained Principal Components for K-means, K-medoids')
    # plt.xlabel('Number of Principal Components')
    # plt.ylabel('Clustering Score')
    # print(str(len(k_means_results)))
    # print(str(len(k_medoids_results)))

    # plt.plot(component_list, k_means_results, 'b', label = 'K-means') 
    # plt.plot(component_list, k_medoids_results, 'r', label='K-medoids')
    # plt.legend()
    # plt.show()


    # part 3c: determine ``most discriminative'' and ``least discriminative'' pairs of images
    # np.random.seed(1234)
    # image_pair_results = {}
    # for i in range(19):
    #     for j in range(i, 19):
    #         if i != j:
    #             Xcur, ycur = util.limit_pics(X, y, [i, j], 40)
    #             points = build_face_image_points(Xcur, ycur)
    #             k_medoids_result = kMedoids(points, 2, init='cheat')
    #             k_medoids_score = k_medoids_result.score()
    #             image_pair_results[(i, j)] = k_medoids_score
    
    # best_score = -1
    # worst_score = 100
    # best_pair = None
    # worst_pair = None
    # for key, val in image_pair_results.items():
    #     if val > best_score:
    #         best_pair = key
    #         best_score = val
    #     if val < worst_score:
    #         worst_pair = key
    #         worst_score = val

    # higher score = more different
    # print 'Best score of %.4f, with images %d and %d' % (best_score, best_pair[0], best_pair[1])
    # plot_representative_images(X, y, [best_pair[0], best_pair[1]], title='Most different images')
    
    # # lower score = less different
    # print 'Worst score of %.4f, with images %d and %d' % (worst_score, worst_pair[0], worst_pair[1])
    # plot_representative_images(X, y, [worst_pair[0], worst_pair[1]], title='Most similar images')
    
    ### ========== TODO : END ========== ###


if __name__ == "__main__" :
    main()
