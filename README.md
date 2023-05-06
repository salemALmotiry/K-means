How to Use the KMeans Model

To use the KMeans model, follow these steps:

Step 1: Initialize the KMeans object with the desired number of clusters, as shown below:

km = KMeans(4)  # Initialize KMeans object with 4 clusters

Step 2: Specify the path of the data file or set of files. We recommend using an absolute path to avoid any issues.

Fpath = r"clustering-data"  # Specify the path of the data file or set of files

Step 3: Use the data_handler method to load and preprocess the data. This method requires five parameters: mode, start_col, end_col, delimiter, and path. The mode parameter specifies whether you are working with a single file or a set of files. The start_col and end_col parameters allow you to specify which columns of the data you want to use. The delimiter parameter specifies the character that separates the columns of the data. Finally, the path parameter specifies the path of the data file or set of files.

```data = km.data_handler(mode=0, start_col=1, end_col=-1, delimiter=' ', path=Fpath) ```

Step 4: Initialize the centroids to start the clustering process using the randomCentroids method.

```init = km.randomCentroids(data_length=data.shape[0])```

Step 5: If you want to use PCA or L2-norm to preprocess the data, use the corresponding methods pca or l2_norm.
```
data = clu.pca(data, 2)  # Use PCA to reduce the data to 2 dimensions
data = km.l2_norm(data)  # Use L2-norm to normalize the data
```
Step 6: Finally, run the KMeans algorithm using the KMeans method. This method requires four parameters: data, measure_distance, init_centroids, and iteration. The data parameter is the preprocessed data. The measure_distance parameter specifies the distance metric to be used (Euclidean distance or Manhattan distance). The init_centroids parameter is the initial set of centroids. The iteration parameter specifies the number of iterations to run the algorithm.

`cluster, centroids = km.KMeans(data=data, measure_distance=km.Euclidean_distance, init_centroids=init, iteration=10)`

Step 7: Print the output.

print(cluster, centroids)
```
cluster = 
[1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0
 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1 3 3 3 3 3 3 3 3 3
 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 0 0 2 0 0 1 0 0 0 2 0
 0 0 0 0 2 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 2 0 0
 0 0 0 0 0 0 0 0 0 0 2 0 2 2 2 2 2 2 2 2 2 2 2 2 0 2 2 0 2 2 2 2 2 2 2 2 2
 0 2 2 2 2 0 2 2 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 2 2]

 centroids = 
  [[ 2.28871274  0.23353724]   
 [ 0.83905403  2.99469221]
 [ 3.66994304 -1.60863114]
 [-2.46598582 -0.43026126]]
```
Additional Functions for KMeans Model

Apart from the basic KMeans functionality, there are some additional functions that you can use:

Evaluating Function: This function can be used to calculate the precision, recall, and F-score of the clustering results. However, it can only be used if you pass a set of files at the beginning.
```P, R, F = km.Evaluating(cluster)```

ExtractFile Function: This function can be used to extract labeled files from the preprocessed data.
km.extractFile(data, cluster)

KMeans_Step_by_Step Function: This function provides a visualization of the clustering process. You can use it as follows:
`Clusters = km.KMeans_Step_by_Step(original_data=data, measure_distance=km.Euclidean_distance, iteration=10)`

Note that you should not pass PCA data to this function. It will automatically reduce the data to 2D features using PCA.

Multi-K Function: If you want to try out multiple values of K, you can use the following code:
```
km = KMeans(0)  # Initialize KMeans object with 0 clusters
Fpath = r"clustering-data"  # Specify the path of the data file or set of files
P, R, F = [], [], []
data = km.data_handler(0, 1, -1, ' ', Fpath)  # Load and preprocess the data
data = km.pca(data, 2)  # Use PCA to reduce the data to 2 dimensions

for i in range(1, 10):
    km.K = i  # Change K for every iteration
    init = km.randomCentroids(data_length=data.shape[0])  # Initialize the centroids
    cluster, centroids = km.KMeans(data=data, measure_distance=km.Manhattan_distance, init_centroids=init, iteration=10)  # Run KMeans
    km.isStillMoving = True  # Reset the "isStillMoving" flag
    Pi, Ri, Fi = km.Evaluating(cluster)  # Calculate precision, recall, and F-score
    P.append(Pi)
    R.append(Ri)
    F.append(Fi)

plt.plot(P, label="Precision")
plt.plot(R, label="Recall")
plt.plot(F, label="F-score")
plt.ylabel("P.R.F")
plt.xlabel("K")
plt.legend()
plt.show()
```
