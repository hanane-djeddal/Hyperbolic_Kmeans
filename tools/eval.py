import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
import math
import random
from decimal import *

def get_CBBU(df,P):
    maxgain=0
    mingain=100000
    gains=[]
    agg=0
    for i in P.keys():
        agg=0
        for j in P[i]:
            agg+=df.iloc[j][2]
        gains.append(agg)
    return max(np.array(gains))
def gain(df,P,cbbu):
    maxgain=0
    mingain=1e20
    gains=[]
    agg=0
    for i in P.keys():
        agg=0
        for j in P[i]:
            agg+=df.iloc[j][2]
        g=agg/cbbu
        gains.append(g)
        if (g>maxgain):
            maxgain=g
        if(g<mingain):
            mingain=g
    moygain=np.mean(np.array(gains))
    return maxgain,mingain,moygain
#Fonctions d'usage
def get_min_centre(centres, K):
    if(len(centres)<K):
        return 0
    else:
        return min(centres['Coord_Z'])
def get_variance(centres, K):
    liste=list(centres['Coord_Z'])
    somme=0
    for i in range(len(centres)-1):
        for j in range(i+1,len(centres)):
            somme+=(liste[i]-liste[j])**2
    return (2/K/(K-1))*somme

def get_variance_diametre(l_affectation,df):
    somme=0
    K=len(l_affectation)
    for i in range(K-1):
        cluster=l_affectation[i]
        dist=get_largeur_cluster(cluster,df)
        for j in range(i+1,K):
            cluster2=l_affectation[j]
            dist2=get_largeur_cluster(cluster2,df)
            somme+=(dist-dist2)**2
    return (2/K/(K-1))*somme

def get_largeur_cluster(cluster,df):
    dist_max=0
    l=len(cluster)
    for i in range(l-1):
        for j in range(i+1,l):
            dist=distance_euclidienne((df['Coord_X'][cluster[i]],df['Coord_Y'][cluster[i]]),(df['Coord_X'][cluster[i]],df['Coord_Y'][cluster[j]]))

            if(dist>dist_max):
                dist_max=dist
    return dist_max
def get_max_largeur(l_affectation,df):
    dist_max=0
    distances=[]
    for i in range(len(l_affectation)):
        cluster=l_affectation[i]
        dist=get_largeur_cluster(cluster,df)
        distances.append(dist)
        if(dist>dist_max):
            dist_max=dist
    return dist_max


def get_nb_cluster(l_affectation):
    nb=0
    for l in l_affectation:
        if(l!=[]):
            nb+=1
    return nb
def get_largeur_cluster(cluster,df):
    dist_max=0
    l=len(cluster)
    for i in range(l-1):
        for j in range(i+1,l):
            dist=distance_euclidienne((df['Coord_X'][cluster[i]],df['Coord_Y'][cluster[i]]),(df['Coord_X'][cluster[i]],df['Coord_Y'][cluster[j]]))

            if(dist>dist_max):
                dist_max=dist
    return dist_max
def get_max_largeur(l_affectation,df):
    dist_max=0
    for i in range(len(l_affectation)):
        cluster=l_affectation[i]
        dist=get_largeur_cluster(cluster,df)
        if(dist>dist_max):
            dist_max=dist
    return dist_max
"""def get_util_cost(dfs,l_affectation):
    costc=0
    costs=0
    #Utilisation Cluster
    Uck=[]
    for i in l_affectation.keys():# pour chaque cluster
        c=l_affectation[i]
        if(c!=[]):
            df_ag_c=[]
            # Traffic agrégé du cluster
            for df in dfs:
                t=[df['Coord_Z'][si] for si in c]
                df_ag_c.append(sum(t))

            #Utilisation du cluster
            Uck.append(np.mean(df_ag_c)/max(df_ag_c))
            costc+=max(df_ag_c)
       
    #Utilisation Station
    Usi=[]
    for si in range(len(dfs[0])):# pour chaque station
        t=[df['Coord_Z'][si] for df in dfs]
        Usi.append(np.mean(t)/max(t))            
        costs+=max(t)  
    utilisation=np.mean(Uck)/np.mean(Usi)
    cost=costc/costs  
    return utilisation,cost"""
def get_util_cost(dfs,l_affectation):
    costc=0
    costs=0
    #Utilisation Cluster
    Uck=[]
    for i in l_affectation.keys():# pour chaque cluster
        c=l_affectation[i]
        if(c!=[]):
            df_ag_c=[]
            # Traffic agrégé du cluster
            for df in dfs:
                t=[df['Coord_Z'][si] for si in c]
                df_ag_c.append(sum(t))
            #Utilisation du cluster
            #Uck.append(np.mean(df_ag_c)/max(df_ag_c))
            Uck.append(sum(df_ag_c)/(max(df_ag_c)*len(dfs)))
            costc+=max(df_ag_c)
      
    #Utilisation Station
    Usi=[]
    for si in range(len(dfs[0])):# pour chaque station
        t=[df['Coord_Z'][si] for df in dfs]
        Usi.append(np.mean(t)/max(t))            
        costs+=max(t)  
    utilisation=np.mean(Uck)/np.mean(Usi)
    cost=costc/costs   
    return utilisation,cost  

########################## Adapted for DCCA

""" affichage des clusters """
def display_cluster(P):
    k=0
    lescomp=[]
    plt.title("Clustering DCCA, To =0.24")
    coord_X=[]
    coord_Y=[]
    coord2=[]
    clusters=[]
    for C in P:
        if (C!=[]):
            k+=1
            lescomp.append(DCCA.complementarity(C,CBBU))
            X=np.empty([len(C),2])
            for i in range(len(C)):
                #ligne=df_geo.loc[df_geo['LocInfo']==C[i].id,:]
                #x=ligne.iloc[0,1]
                #y=ligne.iloc[0,2]
                x=C[i].lat
                y=C[i].lng
                X[i][0]=x
                X[i][1]=y
                coord_X.append(x)
                coord_Y.append(y)
                if(not [x,y] in coord2):
                    coord2.append([x,y])
                clusters.append(k)
            plt.scatter(X[:,0],X[:,1])
    return coord2,clusters,k

def util_cost(P,F):
    Uc=[]
    Cc=[]
    Us=[]
    Cs=[]
    for c in P:
        if c!=[]:
            agg=DCCA.aggregationTrafic(c)
            moy=sum(agg)/len(agg)
            mx=max(agg)
            Uc.append(moy/mx)
            Cc.append(mx)
    for l in F:
        moy=sum(l)/len(l)
        mx=max(l)
        Us.append(moy/mx)
        Cs.append(mx)
    U=(sum(Uc)/len(Uc))/(sum(Us)/len(Us))
    C=sum(Cc)/sum(Cs)
    return U,C
def get_max_largeur_DCCA(P):
    dist=0
    for c in P:
        if c!=[]:
            for i in range(len(c)-1):
                x=c[i].lat
                y=c[i].lng
                for j in range(i+1,len(c)):
                    x2=c[j].lat
                    y2=c[j].lng
                    d=math.sqrt((x-x2)**2 + (y-y2)**2)
                    if (d > dist):
                        dist=d
    return dist
def get_k(P):
    n=0
    for c in P:
        if c!=[]:
            n+=1
    return n

