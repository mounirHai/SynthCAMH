# Library of Functions for the OpenClassrooms Multivariate Exploratory Data Analysis Course

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from pandas.plotting import parallel_coordinates
import seaborn as sns
from prince import MCA,PCA
import plotly.express as px
import networkx as nx
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
from kmodes import kprototypes
from sklearn.metrics import silhouette_score
from scipy.spatial import distance




palette = sns.color_palette("bright", 10)
def cluster_profile_cat(df):
    dfc = df.groupby("cluster").agg({ 
        "sexDesc": lambda x: x.value_counts().index[0],
          "ageGrpDesc": lambda x: x.value_counts().index[0],
      #  "referralLength_cat":  lambda x: x.value_counts().index[0],
         "intensityGrp":  lambda x: x.value_counts().index[0],
        "intensity":   "median",
        "cond1axis1Grp_1": lambda x: x.value_counts().index[0]

      
                                    })    #.sort_values(by=["MonthlyCharges"], ascending=False)

   
    return dfc
# helper function: medians and modes for each cluster
def cluster_profile(df):
    dfc = df.groupby("cluster").agg({ 
        "sexDesc": lambda x: x.value_counts().index[0],
        #  "ageGrpDesc": lambda x: x.value_counts().index[0],
         "age": "median",
        # "ageGrpDesc": lambda x: x.value_counts().index[0],
        #"referralLength": "median",
        # "numOfStays": "median",
       "intensity":   "mean",
        "cond1axis1Grp_1": lambda x: x.value_counts().index[0],
        "cond1axis1Grp_2": lambda x: x.value_counts().index[1],
      #  "cond1axis1Grp_3": lambda x: x.value_counts().index[2],
      
                                    })    #.sort_values(by=["MonthlyCharges"], ascending=False)

   
    return dfc
def condense_category(col, min_freq=0.1, new_name='andre'):
    series = pd.value_counts(col)
    mask = (series/series.sum()).lt(min_freq)
    return pd.Series(np.where(col.isin(series[mask].index), new_name, col))

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    """Display correlation circles, one for each factorial plane"""

    # For each factorial plane
    for d1, d2 in axis_ranks: 
        if d2 < n_comp:

            # Initialise the matplotlib figure
            fig, ax = plt.subplots(figsize=(10,10))

            # Determine the limits of the chart
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # Add arrows
            # If there are more than 30 arrows, we do not display the triangle at the end
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (see the doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # Display variable names
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # Display circle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # Define the limits of the chart
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # Display grid lines
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # Label the axes, with the percentage of variance explained
            plt.xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Correlation Circle (PC{} and PC{})".format(d1+1, d2+1))
            plt.show(block=False)
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    '''Display a scatter plot on a factorial plane, one for each factorial plane'''

    # For each factorial plane
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # Initialise the matplotlib figure      
            fig = plt.figure(figsize=(7,6))
        
            # Display the points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # Display the labels on the points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # Define the limits of the chart
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # Display grid lines
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # Label the axes, with the percentage of variance explained
            plt.xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection of points (on PC{} and PC{})".format(d1+1, d2+1))
            #plt.show(block=False)
   
def display_scree_plot(pca):
    '''Display a scree plot for the pca'''

    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("Number of principal components")
    plt.ylabel("Percentage explained variance")
    plt.title("Scree plot")
    plt.show(block=False)

def append_class(df, class_name, feature, thresholds, names):
    '''Append a new class feature named 'class_name' based on a threshold split of 'feature'.  Threshold values are in 'thresholds' and class names are in 'names'.'''
    
    n = pd.cut(df[feature], bins = thresholds, labels=names)
    df[class_name] = n

def plot_dendrogram(Z, names, figsize=(10,25)):
    '''Plot a dendrogram to illustrate hierarchical clustering'''

    plt.figure(figsize=figsize)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    #plt.show()

def addAlpha(colour, alpha):
    '''Add an alpha to the RGB colour'''
    
    return (colour[0],colour[1],colour[2],alpha)

def display_parallel_coordinates(df, num_clusters):
    '''Display a parallel coordinates plot for the clusters in df'''

    # Select data points for individual clusters
    cluster_points = []
    for i in range(num_clusters):
        cluster_points.append(df[df.cluster==i])
    
    # Create the plot
    fig = plt.figure(figsize=(12, 15))
    title = fig.suptitle("Parallel Coordinates Plot for the Clusters", fontsize=18)
    fig.subplots_adjust(top=0.95, wspace=0)

    # Display one plot for each cluster, with the lines for the main cluster appearing over the lines for the other clusters
    for i in range(num_clusters):    
        plt.subplot(num_clusters, 1, i+1)
        for j,c in enumerate(cluster_points): 
            if i!= j:
                pc = parallel_coordinates(c, 'cluster', color=[addAlpha(palette[j],0.2)])
        pc = parallel_coordinates(cluster_points[i], 'cluster', color=[addAlpha(palette[i],0.5)])

        # Stagger the axes
        ax=plt.gca()
        for tick in ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(20)        


def display_parallel_coordinates_centroids(df, num_clusters):
    '''Display a parallel coordinates plot for the centroids in df'''

    # Create the plot
    fig = plt.figure(figsize=(12, 5))
    title = fig.suptitle("Parallel Coordinates plot for the Centroids", fontsize=18)
    fig.subplots_adjust(top=0.9, wspace=0)

    # Draw the chart
    parallel_coordinates(df, 'cluster', color=palette)

    # Stagger the axes
    ax=plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)  

def get_pca_2d(df, predict):

    pca_2d_object = PCA(
    n_components=2,
    n_iter=3,
    rescale_with_mean=True,
    rescale_with_std=True,
    copy=True,
    check_input=True,
    engine='sklearn',
    random_state=42
    )

    pca_2d_object.fit(df)

    df_pca_2d = pca_2d_object.transform(df)
    df_pca_2d.columns = ["comp1", "comp2"]
    df_pca_2d["cluster"] = predict

    return pca_2d_object, df_pca_2d



def get_pca_3d(df, predict):

    pca_3d_object = PCA(
    n_components=3,
    n_iter=3,
    rescale_with_mean=True,
    rescale_with_std=True,
    copy=True,
    check_input=True,
    engine='sklearn',
    random_state=42
    )

    pca_3d_object.fit(df)

    df_pca_3d = pca_3d_object.transform(df)
    df_pca_3d.columns = ["comp1", "comp2", "comp3"]
    df_pca_3d["cluster"] = predict

    return pca_3d_object, df_pca_3d



def plot_pca_3d(df, title = "PCA Space", opacity=0.8, width_line = 0.1):

    df = df.astype({"cluster": "object"})
    df = df.sort_values("cluster")

    fig = px.scatter_3d(df, 
                        x='comp1', 
                        y='comp2', 
                        z='comp3',
                        color='cluster',
                        template="plotly",
                        
                        # symbol = "cluster",
                        
                        color_discrete_sequence=px.colors.qualitative.Vivid,
                        title=title).update_traces(
                            # mode = 'markers',
                            marker={
                                "size": 4,
                                "opacity": opacity,
                                # "symbol" : "diamond",
                                "line": {
                                    "width": width_line,
                                    "color": "black",
                                }
                            }
                        ).update_layout(
                                width = 1000, 
                                height = 800, 
                                autosize = False, 
                                showlegend = True,
                                legend=dict(title_font_family="Times New Roman",
                                            font=dict(size= 20)),
                                scene = dict(xaxis=dict(title = 'comp1', titlefont_color = 'black'),
                                            yaxis=dict(title = 'comp2', titlefont_color = 'black'),
                                            zaxis=dict(title = 'comp3', titlefont_color = 'black')),
                                font = dict(family = "Gilroy", color  = 'black', size = 15))
                      
    
    return fig


def plot_pca_2d(df, title = "PCA Space", opacity=0.8, width_line = 0.1):

    df = df.astype({"cluster": "object"})
    df = df.sort_values("cluster")

    fig = px.scatter(df, 
                        x='comp1', 
                        y='comp2', 
                        color='cluster',
                        template="plotly",
                        # symbol = "cluster",
                        
                        color_discrete_sequence=px.colors.qualitative.Vivid,
                        title=title).update_traces(
                            # mode = 'markers',
                            marker={
                                "size": 8,
                                "opacity": opacity,
                                # "symbol" : "diamond",
                                "line": {
                                    "width": width_line,
                                    "color": "black",
                                }
                            }
                        ).update_layout(
                                width = 800, 
                                height = 700, 
                                autosize = False, 
                                showlegend = True,
                                legend=dict(title_font_family="Times New Roman",
                                            font=dict(size= 20)),
                                scene = dict(xaxis=dict(title = 'comp1', titlefont_color = 'black'),
                                            yaxis=dict(title = 'comp2', titlefont_color = 'black'),
                                            ),
                                font = dict(family = "Gilroy", color  = 'black', size = 15))
                        
        
    return fig  

def get_MCA_3d(df, predict):
    mca = MCA(n_components =3, n_iter = 100, random_state = 101)
    mca_3d_df = mca.fit_transform(df)
    mca_3d_df.columns = ["comp1", "comp2", "comp3"]
    mca_3d_df["cluster"] = predict
    return mca, mca_3d_df

def get_MCA_2d(df, predict):
    mca = MCA(n_components =2, n_iter = 100, random_state = 101)
    mca_2d_df = mca.fit_transform(df)
    mca_2d_df.columns = ["comp1", "comp2"]
    mca_2d_df["cluster"] = predict
    return mca, mca_2d_df

def convert_df_to_graph_network(df):
  '''
  This function converts a dataframe into an edge list and finally
  into a network graph
  '''
  df = df.copy()
  edges_df = pd.DataFrame()
  # create a name for each row
  length = len(df)
  row_names = ['row '+ str(i) for i in range(1, length+1)]

  original_cols = df.columns
  df['row_name'] = row_names

  for col in original_cols:
    col_edge_df = df[['row_name', col]].rename(columns={col:'to'})
    edges_df = pd.concat([edges_df, col_edge_df], axis=0)

  # set the edge weights to one
  edges_df['weight'] = 1
  edges_df = edges_df.groupby(['row_name', 'to']).count().reset_index()
  edges_df.rename(columns={'row_name':'from'}, inplace=True)

  graph = nx.from_pandas_edgelist(edges_df, source='from',
                                  target='to', edge_attr=['weight'])
  
  return graph

def convert_community_output_to_df(dict_):
  '''
  this function converts the outut given by the community detection
  algorithm
  list_ = {'row 123':1,'row 234':2,'feature_4':1,'feature_2':2} -> Rows, Cluster
                                                                  'row 123',1
                                                                  'row 234',2
  '''

  rows = []
  clusters = []
  features = []
  feature_clust = []
  for key, value in dict_.items():

    if 'row' in key:
      rows.append(key)
      clusters.append(value)

    else:
      features.append(key)
      feature_clust.append(value)


  return pd.DataFrame({'Rows':rows,
                       'Clusters':clusters}), pd.DataFrame({'Features':features,
                                                            'Clusters':feature_clust})


def merge_clusters_back_df(df, communites):
  
  df = df.copy()
  length = len(df)
  row_names = ['row '+ str(i) for i in range(1, length+1)]

  df['row_name'] = row_names

  df = df.merge(communites, how='left', left_on='row_name', right_on='Rows')

  return df.drop(['row_name', 'Rows'], axis=1, inplace=False)
def rm_outliers(df,cols):
#https://stackoverflow.com/questions/35827863/remove-outliers-in-pandas-dataframe-using-percentiles
    cols = ['referralLength','numOfStays'] # one or more

    Q1 = df[cols].quantile(0.25)
    Q3 = df[cols].quantile(0.75)
    IQR = Q3 - Q1

    df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df

def evaluate_labeled_clusters(scores,  preds, labels, name='', X=None):

  if X is not None:

    silhouette = silhouette_score(X, preds, metric='euclidean')
    cal_har = calinski_harabasz_score(X, preds)
    dav_bould = davies_bouldin_score(X, preds)

    adj_mut_info = adjusted_mutual_info_score(labels, preds, average_method='arithmetic')
    adj_rand = adjusted_rand_score(labels, preds)

    content = {'Algorithm':name,
               'Silhouette':silhouette,
               'Calinski_Harabasz':cal_har,
               'Davis Bouldin':dav_bould,
               'Adjusted_Mutual_Info':adj_mut_info,
               'Adjusted_Rand_Score':adj_rand}

    scores = scores.append(content, ignore_index = True)

  else:

    adj_mut_info = adjusted_mutual_info_score(labels, preds, average_method='arithmetic')
    adj_rand = adjusted_rand_score(labels, preds)

    content = {'Algorithm':name,
               'Silhouette':np.NaN,
               'Calinski_Harabasz':np.NaN,
               'Davis Bouldin':np.NaN,
               'Adjusted_Mutual_Info':adj_mut_info,
               'Adjusted_Rand_Score':adj_rand}

    scores = scores.append(content, ignore_index = True)


  return scores

def evaluate_clusters(scores,  clusters_r,clusters_s,  name='',X_r=None, X_s=None):



    silhouette_s = silhouette_score(X_s, clusters_s, metric='precomputed')
    cal_har_s = calinski_harabasz_score(X_s, clusters_s)
    dav_bould_s = davies_bouldin_score(X_s, clusters_s)
    silhouette_r = silhouette_score(X_r, clusters_r, metric='precomputed')
    cal_har_r = calinski_harabasz_score(X_r, clusters_r)
    dav_bould_r = davies_bouldin_score(X_r, clusters_r)

    

    content = {'Algorithm':name,
                'Silhouette Real':silhouette_r,
               'Silhouette Synthetic':silhouette_s,
               'Calinski_Harabasz Real':cal_har_r,
               'Calinski_Harabasz Synthetic':cal_har_s,
                'Davis Bouldin Real':dav_bould_r,
               'Davis Bouldin Synthetic':dav_bould_s}

    #scores = scores.append(content, ignore_index = True)
    scores = pd.concat([scores, pd.DataFrame([content])], ignore_index=True)
  


    return scores



## https://medium.com/analytics-vidhya/the-ultimate-guide-for-clustering-mixed-data-1eefa0b4743b
def get_knn_bins(df, cols, bins=5, drop_cols=True):
  
  for col in cols:
    
    kmeans  = KMeans(n_clusters=bins).fit(df[col].to_frame().values.reshape(-1,1))
    results = pd.DataFrame(kmeans.labels_, columns=[col + '_centroid'])

    df = df.reset_index()
    df[col + '_centroid'] = results[col + '_centroid']
  
    knn_bin_df = pd.DataFrame(kmeans.cluster_centers_)
    knn_bin_df = knn_bin_df.astype(int).reset_index()

    temp_df = pd.merge(df[col + '_centroid'],
                       knn_bin_df, 
                       left_on=col + '_centroid',
                       right_on='index',
                       how='left')
    
    # rename empty column header 0 -> column_name value
    temp_df = temp_df.rename(columns={0:col+'_cat'})

    temp_df.loc[:,col+'_cat'] = col + '_' + temp_df[col+'_cat'].astype(str)

    df = pd.concat([df, temp_df[col+'_cat']], axis=1)
    df.drop([col + '_centroid', 'index'], axis=1, inplace=True)


    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.hist(df[col].values, bins=50)
    ax1.set_title('Histogram of ' + col)

    # produce the second bar chart need the centroids and sort values to make bar chart
    # resemble the original distribution.
    Count_Bins = temp_df[col + '_cat'].value_counts().rename_axis('bins').reset_index(name='counts')
    extract_number = Count_Bins.bins.str.extract('([a-zA-Z]+)([^a-zA-Z]+)', expand=True)
    extract_number.columns = ['Text', 'Number']
    Count_Bins['bin_centroid'] = extract_number['Number']
    Count_Bins.sort_values(by=['bin_centroid'], ascending=True, inplace=True)

    ax2.bar(Count_Bins.iloc[:, 0], Count_Bins.iloc[:, 1])
    ax2.set_title('Bar Chart of ' + col + ' Binned')
    ax2.tick_params(labelrotation=90)

    plt.show()
    plt.tight_layout()


  if drop_cols:
    return df.drop(cols, axis=1)

  else:
    return df

# https://codinginfinite.com/silhouette-coefficient-for-k-modes-and-k-prototypes-clustering/
def mixed_distance(a,b,categorical=None, alpha=0.01):
    if categorical is None:
        num_score=kprototypes.euclidean_dissim(a,b)
        return num_score
    else:
        cat_index=categorical
        a_cat=[]
        b_cat=[]
        for index in cat_index:
            a_cat.append(a[index])
            b_cat.append(b[index])
        a_num=[]
        b_num=[]
        l=len(a)
        for index in range(l):
            if index not in cat_index:
                a_num.append(a[index])
                b_num.append(b[index])
                
        a_cat=np.array(a_cat)#.reshape(1,-1)
       
        a_num=np.array(a_num).reshape(1,-1)
        b_cat=np.array(b_cat)#.reshape(1,-1)
        b_num=np.array(b_num).reshape(1,-1)
        #cat_score=kprototypes.matching_dissim(a_cat,b_cat)
        cat_score=distance.hamming(a_cat,b_cat) * len(a_cat)
        num_score=kprototypes.euclidean_dissim(a_num,b_num)
        return cat_score+num_score*alpha
def dm_prototypes(dataset,categorical=None,alpha=0.1):
    #if the input dataset is a dataframe, we take out the values as a numpy. 
    #If the input dataset is a numpy array, we use it as is.
    if type(dataset).__name__=='DataFrame':
        dataset=dataset.values    
    lenDataset=len(dataset)
    distance_matrix=np.zeros(lenDataset*lenDataset).reshape(lenDataset,lenDataset)
    for i in range(lenDataset):
        for j in range(lenDataset):
            x1= dataset[i]
            x2= dataset[j]
            distance=mixed_distance(x1, x2,categorical=categorical,alpha=alpha)
            distance_matrix[i][j]=distance
            distance_matrix[j][i]=distance
    return distance_matrix
def category_distance(dataset):
    #if the input dataset is a dataframe, we take out the values as a numpy. 
    #If the input dataset is a numpy array, we use it as is.
    if type(dataset).__name__=='DataFrame':
        dataset=dataset.values    
    lenDataset=len(dataset)
    distance_matrix=np.zeros(lenDataset*lenDataset).reshape(lenDataset,lenDataset)
    for i in range(lenDataset):
        for j in range(lenDataset):
            x1= dataset[i]#.reshape(1,-1)
            x2= dataset[j]#.reshape(1,-1)
            cat_score=distance.hamming(x1,x2) * len(x1)
            distance_matrix[i][j]=cat_score
            distance_matrix[j][i]=cat_score
    return distance_matrix






