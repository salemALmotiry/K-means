
import os
from os.path import join,isfile
from os import listdir
import imageio
from matplotlib import pyplot as plt
import numpy as np
import math

from numpy import ma



class Kmeans : 
   

    def __init__(self,K_clustering,) -> None:
         self.K = K_clustering
         self.centroids= np.array([])
         self.isStillMoving = True
         self.Clusters = np.array([])
         
    
    def data_handler(self, mode,jump_B,jump_E,delimiter,Path):
        if mode == 0 :  # collect data in set of files as one array 
            Files_data = []
            for path in listdir(Path):
                full_path = join(Path, path)
             
                if isfile(full_path):

                    data =np.genfromtxt(full_path,dtype=str,delimiter=delimiter)
                    fun = lambda x : x[jump_B:jump_E].astype(float)
                    file_data = list(map(fun,data))
                    Files_data.append(file_data)
    
        
            merge_clustering_data_2dArray = np.concatenate(( Files_data[:]))
            """
                megre_clustering_data_2dArray : 

                [[-0.015926  -0.079864  -0.33218   ... -0.023002   0.0039075 -0.035713 ]
                [ 0.47727   -0.91587   -0.2977    ... -0.44699   -0.24957    0.02851  ]
                [-0.33575    0.38897   -0.41929   ... -0.14929   -0.23516    0.039194 ]
                ...
                [-0.36216   -0.5386    -0.66052   ...  0.026223   0.14061   -0.38506  ]
                [-0.14323   -0.31758   -0.39174   ... -0.56286   -0.58964    0.25269  ]
                [-0.58577   -0.37071   -0.12452   ... -0.0054729 -0.84361    0.087304 ]]

            """
            return merge_clustering_data_2dArray
        
        if mode == 1 : # collect data from one file 
              data =np.genfromtxt(Path,dtype=str,delimiter=delimiter)
              data = data[:,jump_B:jump_E].astype(float)
              return data
    
    
    def pca(self,X,k):
        X_meaned = X - np.mean(X , axis = 0)

        cov_mat = np.cov(X_meaned , rowvar = False)
        
        eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
        
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:,sorted_index]
        
        eigenvector_subset = sorted_eigenvectors[:,0:k]
        
        X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
        
        return X_reduced
    

    def Euclidean_distance(self,k , data):
        # sqrt of (x2 - x1 )^2 + (y2 - y1)^2 + (z2 - z1 )^2 ...
        fun = lambda xi : np.sqrt(np.sum(np.square(np.subtract(  k,  xi) ) )   )
        distance = list( map( fun, data ) )
        return distance
        
    def Manhattan_distance(delf,k,data):
         fun = lambda xi :  np.sum(np.absolute(np.subtract(  k,  xi)) ) 
         distance = list( map( fun, data ) )
         return distance

    def l2_norm(self,data):
        fun = lambda xi : np.sqrt(np.sum(     np.square(xi)            ) )
        norm = list(map(fun,data))
        return    np.array( norm)    

    def Clustering(self,distances): 
        # indx = np.zeros(distances.shape)
        fun = lambda x : np.argmin(x)
        indx = np.fromiter( map(fun,distances.T),int)  
        return indx
    
    
    def compute_new_centroids(self,Clusteringg , data ):
       
        new_centroids = np.zeros((self.K ,data.ndim))
        
        for i in  range(self.K):
            
            indices = np.where(Clusteringg == i)

           
            new_centroids[i] =  np.sum(  data[indices,:],axis=1)/ len(indices[0])

      
        if np.array_equal(new_centroids,self.centroids):
            self.isStillMoving = False
        return new_centroids
  
    def randomCentroids (self,data_shape):
        return np.random.choice(data_shape,  size = self.K,replace=False)

    def Kmeans(self, data,init_centroids,itr ):   
        # random_centroids = self.randomCentroids(data_shape=data.shape[0])
        self.centroids = data[init_centroids]
        # print((self.centroids))
        fun = lambda x : self.Euclidean_distance(x,data)
        self.Clusters = np.zeros(data.shape[0])
        for i in range(itr): 
            if self.isStillMoving == False : 
                break          
            distances = list (map(fun,self.centroids))
            distances = np.array(distances)
          
            self.Clusters = self.Clustering(distances)
           
            self.centroids = self.compute_new_centroids(self.Clusters,data)
            # print(self.Clusters)
            
           
        return np.array(self.Clusters),self.centroids

  
  
    def Kmeans_Step_by_Step(self,X,itr):
           
            data = self.pca(X,2)
            init_centroids = self.randomCentroids(data_shape= data.shape[0])
            img =[]

            for i in range(itr):
                Clusters,centroids = self.Kmeans(data,init_centroids,i)
                fig, ax = plt.subplots(figsize=(9,6))
                for ii in range(self.K):
                    clusteri =  data[np.where(Clusters == ii)]
                    
                    ax.scatter(clusteri[:,0], clusteri[:,1], s=30, label='Cluster %s'%(ii+1))
                    ax.scatter(centroids[ii,0],centroids[ii,1], marker="d",color="blue" ,s=120    )

                  
                
                ax.legend()
               
                    
                for ii in range(4):
                    plt.savefig('%s.png'%i)
                    img.append('%s.png'%i)

            
                        

            with imageio.get_writer('mygif.gif', mode='I') as writer:
                for filename in img:
                    image = imageio.imread(filename)
                    writer.append_data(image)
            for im in range(itr):
                os.remove("%s.png"%im)
           
            print("\nfile 1\n",Clusters[0:49],"\n_______________\nfile 2\n",Clusters[49:210],"\n_______________\nfile 3\n",Clusters[210:268],"\n______________\nfile 4",Clusters[268:-1])
            return  Clusters
          
           
           







       
def main():
     

    clu = Kmeans(10)
    # folder path of the data; mode is 0 to collect all data 
    Fpath = r"clustering-data" # <= relative path , i recommend to use absolute path if there any problem
    data = clu.data_handler(0,1,-1,' ',Fpath)


    C  = clu.Kmeans_Step_by_Step(data,10)

if __name__ == "__main__":
    main()
