How to ues Kmeans model 

km = Kmeans(4)   # init kmeans object with cluster number 

path of set file or one file 
Fpath = r"clustering-data" # <= relative path , i recommend to use absolute path if there any problem

```data = km.data_handler(0,1,-1,' ',Fpath) ```

The data_hanlder needs five parameters
1: mode there is two mode , set of file or one file
2-3 : if you want to skip some part from file for example , [sa,10,2,2,3]  here we want the numerical data then simply pass [1,-1]
4: delimiter 
5 : path

Now init the centroids to start 

```init = km.randomCentroids(data_length= data.shape[0])```

if you wish to use pca function or l2-norm 

```data = clu.pca(data,2) # 2 is the number of features ```

```data = km.l2_norm(data)```


we are ready to run the kmeans 

```cluster , centroids =  km.Kmeans(data=data,measure_distance=km.Euclidean_distance ,init_centroids=init,iteration=10)```

you can pass euclidean distance or manhattan distance function

let print output 
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
Now we can use the Evaluating function only if pass set of file at beginning 

```P , R  , F = km.Evaluating(cluster)```

in the end , we can use extractFile function to extract labeled files  

```km.extractFile(data,cluster)```

___________

Extra function 

Kmeans_Step_by_Step function provide a git image of clustering process , how to use

```Clusters= km.Kmeans_Step_by_Step(original_data= data ,measure_distance = km.Euclidean_distance, iteration = 10 )```

do not pass pca data the function will be automatically pca data to 2d features.


if you wish to try multi k try this 
  ```
        km = Kmeans(0)   # init kmeans object with cluster number 
        Fpath = r"clustering-data" # <= relative path , i recommend to use absolute path if there any problem
        P , R, F = [],[],[]
        data = km.data_handler(0,1,-1,' ',Fpath)
        data = km.pca(data,2)

        for i in range(1,10):
            km.K = i # change k evry time 
            
            init = km.randomCentroids(data_length= data.shape[0])
            cluster , centroids = km.Kmeans(data=data,measure_distance=km.Manhattan_distance,init_centroids=init,iteration=10)
            
            km.isStillMoving = True # reset the "isStillMoving"
            Pi , Ri , Fi = km.Evaluating(cluster)
            P.append(Pi)
            R.append(Ri)
            F.append(Fi)
        
        plt.plot(P,label= "Precision")
        plt.plot(R,label= "Recall")
        plt.plot(F,label= "F-score")

        plt.ylabel("P.R.F")
        plt.xlabel("K")
        plt.legend()
        plt.show()
        ```



