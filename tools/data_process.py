import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
import math
import random
from decimal import *


import datetime 
import calendar 
  
def findDay(date): 
    born = datetime.datetime.strptime(date, '%Y-%m-%d').weekday() 
    return (calendar.day_name[born]) 

def construct_data_set(traffic,the_ids):
    d={}
    mean_week={}
    for t in range(len(traffic)):
        id=traffic['CellID'][t]
        if(id in the_ids):
            if(id not in mean_week.keys()):
                mean_week[id]={'Monday':{},'Tuesday':{},'Wednesday':{},'Thursday':{},'Friday':{},'Saturday':{},'Sunday':{}}
            if(id not in d.keys()):
                d[id]={}
            tm=traffic[' TimeSlot'][t]
            date=tm[1:11]
            heur=tm[12:17]
            weekday=findDay(date)
            byteUp=traffic[' ByteUp'][t]
            byteDown=traffic[' ByteDn'][t]
            if(date not in d[id].keys()):
                d[id][date]={}
            if(heur not in mean_week[id][weekday].keys()):
                mean_week[id][weekday][heur]=[0,0,0]
            mean_week[id][weekday][heur]=[mean_week[id][weekday][heur][0]+byteUp,mean_week[id][weekday][heur][1]+byteDown,mean_week[id][weekday][heur][2]+1]
            d[id][date][heur]=(byteUp,byteDown)  
    for id in mean_week:
        for day in mean_week[id]:
            for heur in mean_week[id][day]:
                nb=mean_week[id][day][heur][2]
                mean_week[id][day][heur]=[mean_week[id][day][heur][0]/nb,mean_week[id][day][heur][1]/nb]
    return d, mean_week
            

def get_dataframes(timeslots,date,d,mean_week,loc_lille,the_ids):
    dataFrames=[]
    dict_loc_rrhs={}
    for heurQQ in timeslots:
        supp=[]
        dict_loc_id={}
        dict_loc_traf={}
        les_X=[]
        les_Y=[]
        les_Z=[]
        for i in range(len(loc_lille)):
            k=loc_lille['LocInfo'][i]
            if(k in the_ids):
                l=(loc_lille[' Coord_X'][i],loc_lille[' Coord_y'][i])
                d[k]['loc']=l
                if(l not in dict_loc_rrhs.keys()):
                    dict_loc_rrhs[l]=[]
                dict_loc_rrhs[l].append(loc_lille['LocInfo'][i])
                if(l not in dict_loc_id.keys()):
                    z=0
                    if(mean_week[k][date]!={}):
                        if(heurQQ in mean_week[k][date].keys()):
                            z=mean_week[k][date][heurQQ][1]
                    dict_loc_traf[l]=z
                    dict_loc_id[l]=k
                    les_X.append(loc_lille[' Coord_X'][i])
                    les_Y.append(loc_lille[' Coord_y'][i])
                    les_Z.append(z)
                
        ar = {'Coord_X': les_X, 'Coord_Y': les_Y, 'Coord_Z': les_Z}
        df_3D = pd.DataFrame(ar)
        dataFrames.append(df_3D)
    return dataFrames,dict_loc_rrhs

def get_trafic_per_position(dict_loc_rrhs,time_slot_hour,mean_week):
    week_date=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    pos_mean_week={}
    for pos in dict_loc_rrhs.keys():
        l=set(dict_loc_rrhs[pos])
        pos_mean_week[pos]={'Monday':{},'Tuesday':{},'Wednesday':{},'Thursday':{},'Friday':{},'Saturday':{},'Sunday':{}}
        for date in week_date:
            for l_slot in time_slot_hour:
                for slot in l_slot:
                    a=0
                    for rrh in l:
                        if(slot in mean_week[rrh][date].keys()):
                            a+=mean_week[rrh][date][slot][1]
                    pos_mean_week[pos][date][slot]=a
    return pos_mean_week
                    
def moyenne_heur(dfs):
    X=list(dfs[0]['Coord_X'])
    Y=list(dfs[0]['Coord_Y'])
    Z=[]
    for i in range(len(dfs[0])):
        sumZ=0
        tz=0
        for df in dfs:
            sumZ+=df['Coord_Z'][i]
            if(df['Coord_Z'][i]>0):
                tz+=1
        if(sumZ!=0):
            Z.append(sumZ/tz)
        else:
            Z.append(0)
    ar = {'Coord_X': X, 'Coord_Y': Y, 'Coord_Z': Z}
    return pd.DataFrame(ar)
def get_24_dfs(date,pos_mean_week,time_slot_hour):
    res=[]
    X=[i[0] for i in pos_mean_week.keys()]
    Y=[i[1] for i in pos_mean_week.keys()]
    for slots in time_slot_hour:
        df_hour=[]
        for s in slots:
            Z=[]
            for t_week in pos_mean_week.values():
                Z.append(t_week[date][s])
            ar = {'Coord_X': X, 'Coord_Y': Y, 'Coord_Z': Z}
            df_hour.append(pd.DataFrame(ar))
        
        res.append(moyenne_heur(df_hour))   
    return res
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

############################# For DCCA

def generate_day_matrix(day):
    #Nr=df_geo_norm.count()[0]  #nombre des RRHs
    train_traffic=traffic_per_day[day]
    RRH_ID=train_traffic['CellID'].unique()
    #Nr=np.size(RRH_ID)  # nbre de RRH dans données trafic < nbre RRHs dans geo données
    r=[]
    F=np.empty([Nr,Nt])
    Fup=np.empty([Nr,Nt])
    for i in range (Nr) :
        #id=RRH_ID[i]
        lesids=id_per_coord[(coord[i][0],coord[i][1])]
        trafic_RRH =[]
        for id in lesids:
            trafic_RRH +=[train_traffic.loc[train_traffic['CellID']==id,:]]
        trafic_RRH=pd.concat(trafic_RRH,axis=0)
        for h in range(Nt):
            #print(     "H= ",h,"/24")
            if (h <10):
                sh='0'+str(h)
            else:
                sh=str(h)
            tu=0
            td=0
            nbr=0
            j=0
            for ind in (trafic_RRH.index.values):
                slot=trafic_RRH.iloc[j]
                if(slot[' TimeSlot'][12:14] == sh):
                    trafic_RRH=trafic_RRH.drop([ind])
                    nbr+=1
                    td+=slot[' ByteDn']
                    tu+=slot[' ByteUp'] 
                else:
                    j+=1
            if(nbr >0):
                F[i][h]=td/nbr
                Fup[i][h]=tu/nbr 
            else :
                F[i][h]=td
                Fup[i][h]=tu
        r.append(DCCA.RRH(lesids[0],i,df_geo_norm.iloc[i,0],df_geo_norm.iloc[i,1],F[i]))
    return F,Fup,r
def generate_day_matrix2(df,df_norm,loc_dict):
    r=[]
    Nr=df[0].shape[0]
    Nt=len(df)
    F=np.empty([Nr,Nt])
    for h in range (len(df)):
        for j in range (len(df[h])):
            F[j][h]=(df_norm[h])['Coord_Z'][j]
    for j in range(len(df[0])):
        r.append(DCCA.RRH(loc_dict[(df[0].iloc[j,0],df[0].iloc[j,1])],j,df[0].iloc[j,0],df[0].iloc[j,1],F[j]))
    return F,r
    

def test_find_day():
    date = '2019-03-20'
    print(findDay(date)) 
    