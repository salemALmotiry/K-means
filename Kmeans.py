
from os.path import join,isfile
from os import listdir,remove
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
import imageio



import pandas as pd

class Kmeans : 
   

    def __init__(self,K_clustering,) -> None:
         self.K = K_clustering
         self.centroids= np.array([])
         self.isStillMoving = True
         self.Clusters = np.array([])
         self.Cluster_range = [0]
         
    
    def data_handler(self, mode,jump_B,jump_E,delimiter,Path):
        temp = 0 
        if mode == 0 :  # collect data in set of files as one array 
            Files_data = []
            for path in listdir(Path):
                full_path = join(Path, path)
                if isfile(full_path):
                    data =np.genfromtxt(full_path,dtype=str,delimiter=delimiter)
                
                    temp+= len(data)
                    self.Cluster_range.append(temp)
                    
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
              fun = lambda x : x[jump_B:jump_E].astype(float)
              file_data = list(map(fun,data))
             
              return np.array(file_data)
    
    
    def pca(self,data,dim):
        data_meaned = data - np.mean(data , axis = 0)

        cov_mat = np.cov(data_meaned , rowvar = False)
        
        eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
        
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvectors = eigen_vectors[:,sorted_index]
        
        eigenvector_subset = sorted_eigenvectors[:,0:dim]
        
        data_reduced = np.dot(eigenvector_subset.transpose() , data_meaned.transpose() ).transpose()
        
        return data_reduced
    
    def randomCentroids (self,data_length):
        return np.random.choice(data_length,  size = self.K,replace=False)

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
      
        fun = lambda x : np.argmin(x)

        indx = np.fromiter( map(fun,distances.T),int)  
        return indx
        
    def compute_new_centroids(self,Clusteringg , data ):
        try:
            dim = data.shape[1]
        except:
          
            dim = 1
        new_centroids = np.zeros((self.K ,dim))

        for i in  range(self.K) :
            
            indices = np.where(Clusteringg == i)
            try : 
               new_centroids[i] =  np.sum(  data[indices,:] ,axis=1)/ len(indices[0])
            except : 
               new_centroids[i] =  np.sum(  data[indices])/ len(indices[0])
      
        if np.array_equal(new_centroids,self.centroids):
            self.isStillMoving = False
        return new_centroids
  
    
    def Kmeans(self, data,measure_distance,init_centroids,iteration ):   
        
        self.centroids = data[init_centroids]
        # print(self.centroids, " \n_______")
     
        fun = lambda x : measure_distance(x,data)

        self.Clusters = np.zeros(data.shape[0])
        
        for i in range(iteration): 
            if self.isStillMoving == False : 
                break          
            # print(self.centroids,"self\n")
            distances = list (map(fun,self.centroids))
      
            distances = np.array(distances)
           
        
            self.Clusters = self.Clustering(distances)
           
            self.centroids = self.compute_new_centroids(self.Clusters,data)
            
            
           
        return np.array(self.Clusters),self.centroids

  
  
    def Kmeans_Step_by_Step(self,original_data,measure_distance,iteration):
           
            data = self.pca(original_data,2)
            init_centroids = self.randomCentroids(data_length= data.shape[0])
            img =[]

            for i in range(iteration):
                Clusters,centroids = self.Kmeans(data,measure_distance,init_centroids,i)
                fig, ax = plt.subplots(figsize=(9,6))
                for j in range(self.K):
                    clusteri =  data[np.where(Clusters == j)]
                    
                    ax.scatter(clusteri[:,0], clusteri[:,1], s=30, label='Cluster %s'%(j+1))
                    ax.scatter(centroids[j,0],centroids[j,1], marker="d",color="blue" ,s=120    )

                  
                
                ax.legend()
               
                    
                for ii in range(4):
                    plt.savefig('%s.png'%i)
                    img.append('%s.png'%i)

            
                        

            with imageio.get_writer('ClustersPlot.gif', mode='I') as writer:
                for filename in img:
                    image = imageio.imread(filename)
                    writer.append_data(image)
            for im in range(iteration):
                remove("%s.png"%im)
           
            return  Clusters
          
    
        
    def Evaluating(self,clusters):
        TP_FP = 0 
        TP = 0 
        temp_list = []
        Cluster_frequency = []
        
        for i in range(len(self.Cluster_range)-1):
            temp_list.append(  clusters[self.Cluster_range[i]:self.Cluster_range[i+1] ] )

            TP_FP+=math.comb(temp_list[i].shape[0],2)
            
            unique, counts = np.unique(temp_list[i], return_counts=True)
            Cluster_frequency.append( np.asarray((unique, counts)).T)
        
        total = clusters.shape[0] * ( clusters.shape[0]  - 1) / 2
        totalN = total - TP_FP
        
        x = [] 
        for i in Cluster_frequency :
            for j in i :
                x.append(j)
                if j[1] > 1 :
                    TP += math.comb(int(j[1]),2)
    
       
        temp_list =  np.array(x)
    
        FP = TP_FP - TP 
        #FN code 
        mul = 0
    
        for i in range(self.K):
            cluster_i = np.where(temp_list[:,0] == i )
            cluster_i = temp_list[cluster_i]

            while(True):
                    if cluster_i.shape[0] == 0:
                        break
                    temp = cluster_i[0,1] #Cluster frequency 
                    cluster_i = np.delete(cluster_i,0,0)
                    sum = 0 
                
                    for j in cluster_i : 
                      
                        sum+=j[1]

                    mul += temp*sum 
       
        FN = mul
        TN = totalN - FN

        P = TP /(TP+FP)
        R = TP/(TP+FN)
        F = 2*(P*R)/(P+R)

        return P,R,F

    
    def extractFile (self,data , clusters):

        df = pd.DataFrame(data)
        df["cluster"] = clusters
        if len(self.Cluster_range )>1:
             for i in range (len(self.Cluster_range)-1 ): 
                    df[self.Cluster_range[i]:self.Cluster_range[i+1] ].to_csv("C%s"%i,index=None)
        else : 
            df.to_csv("data",index=None)


            
    
    
        








       
def main():

  
        km = Kmeans(0)   # init kmeans object with cluster number 
        Fpath = r"clustering-data" # <= relative path , i recommend to use absolute path if there any problem
        P , R, F = [],[],[]
        data = km.data_handler(0,1,-1,' ',Fpath)
        data = km.pca(data,2)

        for i in range(1,10):
            km.K = i
            
            init = km.randomCentroids(data_length= data.shape[0])
            cluster , centroids = km.Kmeans(data=data,measure_distance=km.Manhattan_distance,init_centroids=init,iteration=10)
            km.isStillMoving = True
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

            
            
        
      
    
   
    

    


         
        

        
 

 
    



if __name__ == "__main__":
    main()
  
