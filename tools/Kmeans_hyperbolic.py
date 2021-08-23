import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
import math
import random
from decimal import *


def normalisation(df):
    col=df.columns
    ar={}
    for i in range(len(col)):
        
        colonne= np.array(list(df[col[i]]))
        max_X=max(colonne)
        min_X=min(colonne)
        if((max_X-min_X)!=0):
            max_X=max(colonne)+1
            min_X=min(colonne)-1
            if(col[i]=='Coord_Z'):
                colonne=(colonne-min_X)/(max_X-min_X)
            else:
                colonne=(colonne-min_X)/(max_X-min_X)
        ar[col[i]]=colonne
        
    return pd.DataFrame(ar)

def get_df_rd(dataFrames_nor):
    dataFrames_nor_inv=[]
    for dataF in dataFrames_nor:
        les_Z_inv=1-np.array(dataF['Coord_Z'])
        les_X_nor=np.array(dataF['Coord_X'])
        les_Y_nor=np.array(dataF['Coord_Y'])
        ar = {'Coord_X': les_X_nor, 'Coord_Y': les_Y_nor, 'Coord_Z': les_Z_inv}
        df_3D_nor_inv = pd.DataFrame(ar)
        dataFrames_nor_inv.append(df_3D_nor_inv)
    return dataFrames_nor_inv

def centroide(df):
    return df.mean(axis = 0)

def distance_euclidienne(p1,p2):
    a=(p1[0]-p2[0])**2
    b=(p1[1]-p2[1])**2
    return math.sqrt(a+b)
def distance_euclidienne_3d(p1,p2):
    a=(p1[0]-p2[0])**2
    b=(p1[1]-p2[1])**2
    c=(p1[2]-p2[2])**2
    return math.sqrt(a+b+c)
def distance_euclidienne_denom(a,b,minx,maxx,miny,maxy):
    x1=a[0]*(maxx-minx)+minx
    x2=b[0]*(maxx-minx)+minx
    y1=a[1]*(maxy-miny)+miny
    y2=b[1]*(maxy-miny)+miny
    return distance_euclidienne((x1,y1),(x2,y2))
def distance_hyper_gama(coor1,coor2,gama):
    t=2*coor1[2]*coor2[2]
    d=(((coor1[0]-coor2[0])**2)+((coor1[1]-coor2[1])**2)) 
    res=(gama*(d/t))+(0.5*((coor1[2]/coor2[2])+(coor2[2]/coor1[2])))
    return np.arccosh(res)
def dist_vect2(X,Y,gama):
    res=distance_hyper_gama(X,Y,gama)
    return res
def dist_vect_geo(X,Y):
    res=distance_euclidienne(X,Y)
    return res
def dist_vect_3d(X,Y):
    res=distance_euclidienne_3d(X,Y)
    return res


def inertie_cluster(df,gama):
    c=centroide(df)
    som=0
    for i in range(len(df)):
        x=df.iloc[i]
        som+=dist_vect2(x,c,gama)**2
    return som
def inertie_cluster_geo(df):
    c=centroide(df)
    som=0
    for i in range(len(df)):
        x=df.iloc[i]
        som+=dist_vect_geo(x,c)**2
    return som
def inertie_cluster_3d(df):
    c=centroide(df)
    som=0
    for i in range(len(df)):
        x=df.iloc[i]
        som+=dist_vect_3d(x,c)**2
    return som

def initialisation(K,df): #Initialisation avec comme valeures aleatoires des valeur générées entre 0-1 
    colonnes=df.columns
    m={}                # A utiliser avec la table normalisée seulement
    for c in colonnes:
        points=[]
        for i in range(K):
            points.append(random.uniform(0,1))
        m[c]=points
    return pd.DataFrame(m)

def initialisation2(K,df): #Initialisation avec comme valeures k rrhs
    colonnes=df.columns
    l=df.index.values
    ind=np.random.choice(l,K,replace=False)
    m={}              # A utiliser avec la table normalisée seulement
    for k in ind :
        c=df.iloc[k]
        for col in colonnes:
            if col in m.keys():
                m[col].append(c[col])
            else:
                m[col]=[c[col]]  
    return pd.DataFrame(m)

def plus_proche(x,df_C,gama):
    dis=1e+100
    centre='betise'
    
    for i in range(len(df_C)):
        c=df_C.iloc[i]
        d=dist_vect2(x,c,gama) 
        if(dis>=d):
            dis=d
            centre=i
    return centre
def plus_proche_geo(x,df_C):
    dis=1e+100
    centre='betise'
    
    for i in range(len(df_C)):
        c=df_C.iloc[i]
        d=dist_vect_geo(x,c) 
        if(dis>=d):
            dis=d
            centre=i
    return centre
def plus_proche_3d(x,df_C):
    dis=1e+100
    centre='betise'
    
    for i in range(len(df_C)):
        c=df_C.iloc[i]
        d=dist_vect_3d(x,c) 
        if(dis>=d):
            dis=d
            centre=i
    return centre


def affecte_cluster(df_app, df_C,gama):
    d={}
    l=df_app.index.values
    for i in l:
        x=df_app.loc[i]
        c= plus_proche(x,df_C,gama)
        if(not c in d.keys()):
            d[c]=[]
        d[c].append(i)
    for k in range(len(df_C)):
        if(not k in d.keys()):
            d[k]=[]
    return d
def affecte_cluster_geo(df_app, df_C):
    d={}
    l=df_app.index.values
    for i in l:
        x=df_app.loc[i]
        c= plus_proche_geo(x,df_C)
        if(not c in d.keys()):
            d[c]=[]
        d[c].append(i)
    for k in range(len(df_C)):
        if(not k in d.keys()):
            d[k]=[]
    return d
def affecte_cluster_3d(df_app, df_C):
    d={}
    l=df_app.index.values
    for i in l:
        x=df_app.loc[i]
        c= plus_proche_3d(x,df_C)
        if(not c in d.keys()):
            d[c]=[]
        d[c].append(i)
    for k in range(len(df_C)):
        if(not k in d.keys()):
            d[k]=[]
    return d


def nouveaux_centroides(df_app,d,part):
    m=[]
    i=0
    colonnes=part.columns
    for k, l in d.items():
        i+=1
        X=[]
        for i in l:
            X.append(df_app.loc[i])
        if(X != []):
            s=centroide(pd.DataFrame(X))
            m.append(s)
        else:
            ar = {'Coord_X':[part.iloc[k]['Coord_X']] , 'Coord_Y': [part.iloc[k]['Coord_Y']], 'Coord_Z': [part.iloc[k]['Coord_Z']]}
            s = centroide(pd.DataFrame(ar))
            m.append(s)
    return pd.DataFrame(m)

def inertie_globale(df_app,d,gama):
    sum=0
    for l in d.values():
        X=[]
        for i in l:
            X.append(df_app.loc[i])
        sum+=inertie_cluster(pd.DataFrame(X),gama)
    return sum
def inertie_globale_geo(df_app,d):
    sum=0
    for l in d.values():
        X=[]
        for i in l:
            X.append(df_app.loc[i])
        sum+=inertie_cluster_geo(pd.DataFrame(X))
    return sum
def inertie_globale_3d(df_app,d):
    sum=0
    for l in d.values():
        X=[]
        for i in l:
            X.append(df_app.loc[i])
        sum+=inertie_cluster_3d(pd.DataFrame(X))
    return sum


def kmoyennes(K,df_app,eps,iter_max,gama):
    i=0
    fin=False
    partition=initialisation2(K,df_app)
    dic=affecte_cluster(df_app, partition,gama)
    inertie=inertie_globale(df_app,dic,gama)
    
    while(i<iter_max and fin==False):
        partition=nouveaux_centroides(df_app,dic,partition)
        dic=affecte_cluster(df_app, partition,gama)
        inertie_suiv=inertie_globale(df_app,dic,gama)
        if(abs(inertie_suiv-inertie) < eps):
            fin=True
        i+=1
        inertie=inertie_suiv
    return partition,dic
def kmoyennes_geo(K,df_app,eps,iter_max):
    i=0
    fin=False
    partition=initialisation2(K,df_app)
    dic=affecte_cluster_geo(df_app, partition)
    inertie=inertie_globale_geo(df_app,dic)
    
    while(i<iter_max and fin==False):
        partition=nouveaux_centroides(df_app,dic,partition)
        dic=affecte_cluster_geo(df_app, partition)
        inertie_suiv=inertie_globale_geo(df_app,dic)
        if(abs(inertie_suiv-inertie) < eps):
            fin=True
        i+=1
        inertie=inertie_suiv
    return partition,dic
def kmoyennes_3d(K,df_app,eps,iter_max):
    i=0
    fin=False
    partition=initialisation2(K,df_app)
    dic=affecte_cluster_3d(df_app, partition)
    inertie=inertie_globale_3d(df_app,dic)
    
    while(i<iter_max and fin==False):
        partition=nouveaux_centroides(df_app,dic,partition)
        dic=affecte_cluster_3d(df_app, partition)
        inertie_suiv=inertie_globale_3d(df_app,dic)
        if(abs(inertie_suiv-inertie) < eps):
            fin=True
        i+=1
        inertie=inertie_suiv
    return partition,dic


def get_close_centeroid_2(c,df,radios):
    res=[]
    les_distances=[]
    les_distances_denom=[]
    l=df.index.values
    for i in l: 
        dist=distance_euclidienne((df['Coord_X'][i],df['Coord_Y'][i]),(c[0],c[1]))
        dist_denom=distance_euclidienne_denom((df['Coord_X'][i],df['Coord_Y'][i]),(c[0],c[1]),647675,654456,2624074,2628544)
        if(dist<radios):
            res.append([df['Coord_X'][i],df['Coord_Y'][i],df['Coord_Z'][i]])
            les_distances.append(dist)
            les_distances_denom.append(dist_denom)
            
    if(len(res)>1):
        #print("erreur taille : ", len(res)) 
        les_distances.sort()
        rd=les_distances[1]
        rdn=les_distances_denom[1]
    if(len(res)==0):
        #print("erreur taille : ", len(res))
        res=[[0,0,0]]
        rd=0
        rdn=0
    if(len(les_distances)==1):
        #print("PARFAIT : ", len(res))
        rd=les_distances[0]
        rdn=les_distances_denom[0]
    return res[0],rd,rdn
def kmoyennes_robuste_2(k,dfs,eps,iter_max,gama,radius):
    res=[]
    radius_possible=[]
    radius_possible_denom=[]
    centre_opt=[]
    for dataFrame in dfs:
        les_centres, l_affectation = kmoyennes(k, dataFrame, eps, iter_max,gama)
        res.append([les_centres, l_affectation])
    #moyenner les centres  
    for j in range(len(res[0][0])):
        long=1
        val=np.array([res[0][0]['Coord_X'][j],res[0][0]['Coord_Y'][j],res[0][0]['Coord_Z'][j]])
        for i in range(1,len(res)):
            v,r,rn=get_close_centeroid_2((res[0][0]['Coord_X'][j],res[0][0]['Coord_Y'][j]),res[i][0],radius)
            if(v!=[0,0,0]):
                long+=1
                val+=v
                radius_possible.append(r)
                radius_possible_denom.append(rn)
        val=val/long 
        centre_opt.append(list(val))
    tmp = {'Coord_X': np.array(centre_opt)[:,0], 'Coord_Y': np.array(centre_opt)[:,1], 'Coord_Z': np.array(centre_opt)[:,2]}
    p = pd.DataFrame(tmp)
    dic=affecte_cluster(dfs[0], p,gama)
    return p,dic,min(radius_possible),min(radius_possible_denom)


###########################New version 
def get_close_centeroid(c,df):
    res=[]
    dist_min=1000
    l=df.index.values
    for i in l: 
        dist=distance_euclidienne((df['Coord_X'][i],df['Coord_Y'][i]),(c[0],c[1]))
        dist_denom=distance_euclidienne_denom((df['Coord_X'][i],df['Coord_Y'][i]),(c[0],c[1]),647675,654456,2624074,2628544)
        if(dist<dist_min):
            dist_min=dist
            res=[df['Coord_X'][i],df['Coord_Y'][i],df['Coord_Z'][i]]
    return res
def kmoyennes_robuste_3(k,dfs,eps,iter_max,gama):
    res=[]
    centre_opt=[]
    for dataFrame in dfs:
        les_centres, l_affectation = kmoyennes(k, dataFrame, eps, iter_max,gama)
        res.append(les_centres)
        
    #moyenner les centres  
    for j in range(len(res[0])):
        long=1
        val=np.array([res[0]['Coord_X'][j],res[0]['Coord_Y'][j],res[0]['Coord_Z'][j]])
        for i in range(1,len(res)):
            v=get_close_centeroid((res[0]['Coord_X'][j],res[0]['Coord_Y'][j]),res[i])
            if(v!=[0,0,0]):
                long+=1
                val+=v
        val=val/long 
        centre_opt.append(list(val))
    tmp = {'Coord_X': np.array(centre_opt)[:,0], 'Coord_Y': np.array(centre_opt)[:,1], 'Coord_Z': np.array(centre_opt)[:,2]}
    p = pd.DataFrame(tmp)
    dic=affecte_cluster(dfs[0], p,gama)
    return p,dic












