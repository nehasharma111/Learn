from PyPDF2 import PdfFileMerger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import statistics as st
from numpy.linalg import inv
from scipy.stats import pearsonr
from scipy import stats
from scipy.stats import t
import scipy 
import openpyxl
import os
from os import path
import datetime 
from matplotlib.dates import (YEARLY, DateFormatter, rrulewrapper, RRuleLocator, drange)
import six
import time
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

from pmdarima.utils import c, diff
import pmdarima as pm

from reportlab.lib.styles import ParagraphStyle as PS
from reportlab.platypus import PageBreak
from reportlab.platypus.paragraph import Paragraph
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.platypus.frames import Frame
from reportlab.lib.units import cm, mm, inch
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import *

from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                ParagraphAndImage, Image, Spacer, Table, PageBreak)
from reportlab.platypus import *
from reportlab.lib.styles import ParagraphStyle as PS
from reportlab.platypus.tableofcontents import (TableOfContents, SimpleIndex)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFont
from reportlab.lib.colors import Color
import hashlib
import random
import collections
from functools import partial
from sklearn.impute import SimpleImputer
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 19:32:01 2020

@author: NSharma
"""

# Takes the 'Time_Period' of the data as input, used for changing time index to 'Month-Year'
def TimePeriod(TIME,pred = 0):   
    T = len(TIME) + pred
    con,time = TIME[0].split(" ")
    start = time[4:6]+'/'+'1'+'/'+time[0:4]
    time_index = pd.period_range(start, freq='M', periods=T)  # Creates vector with time-stamps from the first till last month
    return time_index      

# Takes the 'Time_Period' of the data as input, adds time index for predicted months
def TimePeriodPrediction(TIME,start_date,n_pred=0):  
    T = len(TIME) + n_pred
    con,time = start_date.split(" ")
    start = time[4:6]+'/'+'1'+'/'+time[0:4]
    time_index = pd.period_range(start, freq='M', periods=T)
    return time_index   


# Function, creates a dataframe of subsets
# - df: data before creating subsets
# - sub1: subset the dataframe df based on layer1 (e.g. bricks)
# - sub2: subset the dataframe layer1 based on layer2 (e.g. molecules)
# - sub2_name: the specific group within layer2 that you want to create a subset from
# - sub3 and sub4 including names are the same as sub2
# - target: the values that we want to analyse within the subsets (e.g. 'Gram')
# - NOTE!!!!!!: If in any of the bricks there are missing data, do not use this function to  subset the data, use function Complete_subset!!!!
def Data_sub_cu(df,sub1,target,sub2='',sub2_name='',sub3='',sub3_name='',sub4='',sub4_name=''):
    
    sub1_names = df[sub1].unique()     # Categorize the Bricks

    sub = []
    df_single = []
    df_rest = []
    
    if sub2 != '':
        if type(sub2_name) == str:
            for i in range(len(sub1_names)):
                sub.append(df[df[sub1] == sub1_names[i]])
                df_single.append((sub[i])[(sub[i])[sub2] == sub2_name])
                df_rest.append((sub[i])[(sub[i])[sub2] != sub2_name])
        elif type(sub2_name) == list:
            for i in range(len(sub1_names)):
                sub.append(df[df[sub1] == sub1_names[i]])
                df_single.append((sub[i])[(sub[i])[sub2].isin(sub2_name)])
                df_rest.append((sub[i])[~(sub[i])[sub2].isin(sub2_name)])
    
    if sub3 != '':
        if type(sub3_name) == str:
            for i in range(len(sub1_names)):
                df_single[i] = (df_single[i])[(df_single[i])[sub3] == sub3_name]
                df_rest[i] = (df_rest[i])[(df_rest[i])[sub3] != sub3_name]
        elif type(sub3_name) == list:
            for i in range(len(sub1_names)):
                df_single[i] = (df_single[i])[(df_single[i])[sub3].isin(sub3_name)]
                df_rest[i] = (df_rest[i])[~(df_rest[i])[sub3].isin(sub3_name)]
                
    if sub4 != '':
        if type(sub4_name) == str:
            for i in range(len(sub1_names)):
                df_single[i] = (df_single[i])[(df_single[i])[sub4] == sub4_name]
                df_rest[i] = (df_rest[i])[(df_rest[i])[sub4] != sub4_name]
        elif type(sub4_name) == list:
            for i in range(len(sub1_names)):
                df_single[i] = (df_single[i])[(df_single[i])[sub4].isin(sub4_name)]
                df_rest[i] = (df_rest[i])[~(df_rest[i])[sub4].isin(sub4_name)]
                
#    df_single_gram = pd.DataFrame(df_single[0].groupby('TIME_PERIOD').sum()[target])
#    df_rest_gram = pd.DataFrame(df_rest[0].groupby('TIME_PERIOD').sum()[target])
    TIME = sorted(df['TIME_PERIOD'].unique())      # Sort the time
    index=TIME
    columns=sorted(df[sub1].unique())
    df_single_gram=pd.DataFrame(index=index, columns=columns)
    df_rest_gram=pd.DataFrame(index=index, columns=columns)
    for i in range(len(sub1_names)):
        df_single_gram[sub1_names[i]] = df_single[i].groupby('TIME_PERIOD').sum()[target]     # aggregate Gram Astellas
        df_rest_gram[sub1_names[i]] = df_rest[i].groupby('TIME_PERIOD').sum()[target]     # aggregate Gram not Astellas

    df_all_gram = pd.DataFrame(df_single_gram.sum(axis=1))
    
    if len(df_single_gram.index) < 36 or len(df_rest_gram.index) < 36 or len(df_all_gram.index) < 36:
        print('Time index needs to be revised')
    else:   
#        df_single_gram.index = TimePeriod(df_single_gram.index).strftime('%Y-%m')
#        df_single_gram.index.name = 'Time'
#        df_rest_gram.index = df_single_gram.index
        df_all_gram.index = df_single_gram.index
    
#    df_single_gram.columns = sub1_names
#    df_rest_gram.columns = sub1_names
    df_all_gram.columns = ['Nederland']
    return df_single_gram, df_rest_gram, df_all_gram


# Calculate the first difference (t - t-1) and its relative difference 
# single = 0 (default), change to 1 if dataframe only has a single column, or else it will give errors
# name = "" (default)
def First_difference(df,single = 0, name = ""):  
    diff = df[1:].copy()*0
    relative_diff = df[1:].copy()*0
    if single == 1:
        if type(df) == pd.core.frame.DataFrame:
            for i in range(len(df.index)):
                diff.iloc[i] = df.iloc[i+1] - df.iloc[i]
                relative_diff.iloc[i] = diff.iloc[i]/df.iloc[i]
        else:           
            for i in range(len(df)-1):
                diff[i] = df[i+1] - df[i]
                relative_diff[i] = (df[i+1] - df[i])/df[i]
         
    if single == 0: 
        for i in range(len(df.index)-1):
            for j in range(len(df.columns)):
                diff.iloc[i,j] = df.iloc[i+1,j] - df.iloc[i,j]
                relative_diff.iloc[i,j] = (df.iloc[i+1,j] - df.iloc[i,j])/df.iloc[i,j]
    
    return diff, relative_diff


# Count the number of observations crossing each threshold
# df is the dataframe
# crit_neg is a list of negative thresholds (must be from smalles to biggest)
# crit_pos is a list of positive thresholds (must be from smalles to biggest)
# cutoff = 0 (defautlt), change cutoff to a negative number to pick the previous couple of months (-6 for last 6 months)
def Count(df,crit_neg,crit_pos,cutoff=0):    
    if cutoff != 0:
        df_crit = Critical(df[cutoff:],crit_neg,crit_pos)
    else:
        df_crit = Critical(df,crit_neg,crit_pos)
        
    count_neg = np.zeros((len(crit_neg),len(df.columns)))
    count_pos = np.zeros((len(crit_neg),len(df.columns)))
    for i in range(len(crit_neg)):
        count_neg[i] = df_crit[df_crit == crit_neg[i]].count()
    for i in range(len(crit_pos)):
        count_pos[i] = df_crit[df_crit == crit_pos[i]].count()
        
    count_neg = pd.DataFrame(count_neg)
    count_neg.columns = df.columns
    count_neg.index = crit_neg
    count_pos = pd.DataFrame(count_pos)
    count_pos.columns = df.columns
    count_pos.index = crit_pos
    count = count_pos.append(count_neg).sort_index(ascending=False)
    count.index.name = 'Crit'
    
    return df_crit, count

def Critical(df,crit_neg,crit_pos):      # Assign a number based on the max threshold crossed
    crit_over = np.abs(df.copy()*0)
    
    for i in range(len(crit_neg)):
        crit_over[df <= crit_neg[i]] = crit_neg[i]
    for i in range(len(crit_pos)):
        crit_over[df >= crit_pos[i]] = crit_pos[i]

    return crit_over


# Calculate top bricks based on 'Gram'
# cutoff needs to be a negative number (cutoff = -6 to calculate based on the previous 6 months)
def Top(df,cutoff):
    df_old_sum = df[cutoff-1:-1].sum()
    df_new_sum = df[cutoff:].sum()
    TOP_old = df_old_sum.sort_values(ascending=False).index
    TOP_new = df_new_sum.sort_values(ascending=False).index
    
    TOP_same = sorted(list(set(TOP_old[:10]) & set(TOP_new[:10])))    # Intersection between old and new top 10
    TOP_old_diff = np.setdiff1d(TOP_old[:10],TOP_same)          # Brick that got left out of new top 10
    TOP_new_diff = np.setdiff1d(TOP_new[:10],TOP_same)          # Brick that joined new top 10
    
    return TOP_old, TOP_new, TOP_old_diff, TOP_new_diff


# Calculate the ranks of each column in the dataframe based on the total sum
# names is the columns of the dataframe
def Rank(df,top,TOP_old,TOP_new):
    names = df.columns
    df_old_sum = df[cutoff-1:-1].sum()
    df_new_sum = df[cutoff:].sum()
    rank_old_T = pd.DataFrame([range(1,len(names)+1)])
    rank_old = rank_old_T.T
    rank_new_T = pd.DataFrame([range(1,len(names)+1)])
    rank_new = rank_new_T.T
    rank_old.index = df_old_sum.sort_values(ascending=False).index
    rank_old.index.name = 'Brick'
    rank_new.index = df_new_sum.sort_values(ascending=False).index
    rank_new.index.name = 'Brick'
    
    rank = rank_new.copy()
    
    brick = np.full(len(rank.index),'np.nan')
    for i in range(len(rank.index)):
        brick[i] = str(rank.index[i].split(' ')[0])
    rank[0] = brick
    rank[1] = rank_new
    rank[2] = rank_old
    rank.columns = ['brick','new','old']
    rank_top = rank[:top]
    rank_rest = rank[top:]
    rank_left = rank_top.append(rank_rest[rank_rest['old'] <= top])

    return rank, rank_top, rank_rest, rank_left


# Sorts the dataframe based on the TOP list and returns a cut of the previous (cutoff) months
# cutoff is the previous number of months kept in the dataframe (must be a negative number)
# rank is the ranking which orders the dataframe, 1,2,.....
# old = 0 (default), leave if the rank is from the previous month, change to 1 if the rank is from this month
def Cut(df,cutoff,rank,new=0):
    if new == 0:
        df_new = df[rank][:-1]
    else:
        df_new = df[rank]
    df_cut = df_new[cutoff:]
    
    return df_cut


# Creates a dataframe with analyzed information about the dataframe df.
def Dist(df,rank):
    df=df.fillna(0)
    df_old = df[cutoff-1:-1].copy()
    df_new = df[cutoff:].copy()
    
    diff_old = df[cutoff-1:-1].copy()*0
    diff_new = df[cutoff:].copy()*0
         
    for i in range(len(diff_old.index)):
        for j in range(len(diff_old.columns)):
            diff_old.iloc[i,j] = df_old.iloc[i,j] - df_old.mean()[j]
            diff_new.iloc[i,j] = df_new.iloc[i,j] - df_new.mean()[j]
            
    mean_old = df_old.mean()   # Calculate the mean of months -7 to -1 
    mean_new = df_new.mean()   # Calculate the mean of months -6 to 0
    diff_old_abs = np.abs(diff_old)
    diff_new_abs = np.abs(diff_new)        
    
    abs_error = diff_new_abs.mean() - diff_old_abs.mean()
    mean_diff = mean_new - mean_old
    mean_relative_diff =  mean_diff/mean_old
    
    if len(df_new.columns) > 1:
        brick = np.zeros((len(df_new.columns)))
        for i in range(len(df_new.columns)):
            brick[i] = int(df_new.columns[i].split(" ")[0])
    
        dist = pd.DataFrame()
        dist[0] = rank
        dist[1] = brick
        dist[2] = mean_old
        dist[3] = mean_new
        dist[4] = mean_diff
        dist[5] = abs_error
        dist[6] = mean_relative_diff
        dist.columns = ['Rank','Brick','Mean old top','Mean new top','Mean diff','Abs mean diff','% increase']
    
    else:
        dist = pd.DataFrame()
        dist[0] = mean_old
        dist[1] = mean_new
        dist[2] = mean_diff
        dist[3] = abs_error
        dist[4] = mean_relative_diff
        dist.columns = ['Mean old top','Mean new top','Mean diff','Abs mean diff','% increase']
    
    dist_round = np.abs(dist.copy()*0)
    for i in range(len(dist.index)):
        for j in range(len(dist.columns)-1):
            dist_round.iloc[i,j] = int(dist.iloc[i,j])        
        dist_round.iloc[i,-1] = '{:.2%}'.format(dist.iloc[i,-1])
    
    return dist, dist_round


# Prints png of the columns of the dataframe
# outpath is the location of the folder where the prints will be stored
# single = 0 (default), change to 1 if the dataframe has only 1 column or else it will give errors
# TITLE = '' (default), adds a title to the graph
# outpath_add = "" (default), Changes the name of the folder where the graphs will be added
def Brick_to_png_raw(df,single=0,name='',outpath_add = '',TITLE=''):
    if len(outpath_add ) == 0:
        outpath_add= "Raw"
        
    if os.path.exists(outpath_plot + outpath_add) != True:
        os.makedirs(os.path.join("Plots", outpath_add))
        print("Adding new folder: %s"%outpath_add)
        outpath = outpath_plot + outpath_add
    else:
        outpath = outpath_plot + outpath_add
    
    if single == 1:
        brick = df.columns[0]
        
        fig, ax = plt.subplots(figsize=(10,5))
        
        if type(df) == np.ndarray:
            image, = ax.plot(df.index,df,label=brick)
        else:
            image, = ax.plot(df.index,df,label=brick)
        
        ax.xaxis.set_tick_params(rotation=30, labelsize=5)
        ax.set_xlabel('Time')
        if perc == 1:
            ax.set_ylabel('Market Shares')
        else:
            ax.set_ylim(bottom=0)
        ax.set_ylabel('Gram')
        plt.title(TITLE)
        ax.legend()
        fig.savefig(path.join(outpath,"%s.png"%brick),dpi=300)
        plt.close()
        
    
    elif single == 0:
        brick = df.columns
        #brick_num = np.zeros((len(df.columns)))
        #for i in range(len(brick)):
        #    brick_num[i] = int(brick[i].split(" ")[0])

        for i in range(len(brick)):
            fig, ax = plt.subplots(figsize=(10,5))
            image, = ax.plot(df.index,df[brick[i]],label=brick[i])
            ax.xaxis.set_tick_params(rotation=30, labelsize=5)
            ax.set_xlabel('Time')
            ax.set_ylim(bottom=0)
            ax.set_ylabel('Gram')
            plt.title(TITLE)
            ax.legend()
            fig.savefig(path.join(outpath,"%s.png"%brick[i]),dpi=300)
            plt.close()
        fig.clear()
            
    print("done")
    return outpath + "\\"



# Prints png of the past 12 months with a time-series analysis of the past 6 months and a prediction. Rquires dataframe with multiple columns
# out_path add = '' (default), creates a subfolder to store the prints if it is not empty
# conv = 1 (default), increase in case the values of the data is very small (e.g. 0.001))
# TITLE = '' (default), adds a title to the graph
# outpath_add = "" (default), Changes the name of the folder where the graphs will be added
def Brick_edge_png(df,cut=-12,n_pred = 1,outpath_add='',conv=1,TITLE='',perc=0):
    if len(outpath_add ) == 0:
        outpath_add = "Edge"
        
    if os.path.exists(outpath_plot + outpath_add) != True:
        os.makedirs(os.path.join("Plots", outpath_add))
        print("Adding new folder: %s"%outpath_add)
        outpath = outpath_plot + outpath_add
    else:
        outpath = outpath_plot + outpath_add
    sub=[]
    dfi=[]

    sub=pd.DataFrame(df.isna().sum(axis=0))
    sub.columns=['count_na']
    dfi=df
    n_df = len(df.index)
    if range(len(df.columns != 1 )):
        for i in range(len(df.columns)):
            if sub.iloc[i,-1] == n_df:
                dfi.iloc[0,i]=0
    
#    perc = 0
#    if (dfi.iloc[0,0]) < 1:
#        perc = 1
        
    name = df.columns
    df_old = df[TOP_old][:-1].copy()
    df_new = df[TOP_new].copy()
    
    mean_old = df_old[-6:].mean()
    mean_new = df_new[-6:].mean()

    col = ['g--','r--']
    
    for i in range(len(name)):       
        imp = SimpleImputer(missing_values=np.nan,strategy='mean')
        dfi=pd.DataFrame(imp.fit_transform(dfi))
        dfi.columns=df.columns
        dat = dfi.iloc[:,i]*conv
        result = arimamodel(dat)
        dat = dat/conv
        
        mean_old = dat[-7:-1].mean()
        mean_new = dat[-6:].mean()

        n = len(dat.index)
        n_tot = n + n_pred

        prediction, prediction_conf_int = result.predict(n_periods = n_pred,return_conf_int=True)
        prediction, prediction_conf_int = prediction/conv, prediction_conf_int/conv
        in_sample, in_sample_conf_int = result.predict_in_sample(return_conf_int=True)
        in_sample, in_sample_conf_int = in_sample/conv, in_sample_conf_int/conv

        lower_in = in_sample_conf_int[:,0]
        upper_in = in_sample_conf_int[:,1]
        lower_out = prediction_conf_int[:,0]
        upper_out = prediction_conf_int[:,1]
        
        index_pred = TimePeriodPrediction(dat.index,start_date,n_pred=n_pred).strftime('%Y-%m')
        cutoff = cut - n_pred

        in_pred = pd.Series(np.full((n_tot),np.nan))
        in_pred[:n] = in_sample
        dat_pred = pd.Series(np.full((n_tot),np.nan))
        dat_pred[:n],dat_pred[n:] = dat,prediction
        dat1 = pd.Series(np.full((n_tot),np.nan))
        dat1[:n] = dat
        pred1 = pd.Series(np.full((n_tot),np.nan))
        pred1[n-1:n],pred1[n:] = dat.iloc[-1],prediction
        pred2 = pd.Series(np.full((n_tot),np.nan))
        pred2[:n],pred2[n:] = in_sample,prediction.copy()
        lower = pd.Series(np.zeros((n_tot)))
        lower[:n],lower[n:] = lower_in,lower_out
        upper = pd.Series(np.zeros((n_tot)))
        upper[:n],upper[n:] = upper_in,upper_out

        col = ['g--','r--']

        fig, ax = plt.subplots(figsize=(10,5))
        #ax.plot(index_pred[cutoff:],in_pred[cutoff:],'r')
        ax.plot(index_pred[cutoff:],pred2[cutoff:],'r')
        ax.plot(index_pred[cutoff:],dat_pred[cutoff:],'k')
        ax.plot(index_pred[cutoff:],pred1[cutoff:],'g')
        ax.plot(index_pred[cutoff:],dat1[cutoff:],'k',label = dat.name)
        ax.plot(index_pred[cutoff:],lower[cutoff:],'b--',label = 'CI')
        ax.plot(index_pred[cutoff:],upper[cutoff:],'b--')

        #ax.plot(index_pred[cutoff:],mean_new*np.ones((len(index_pred[cutoff:]))),col[0],label = 'new mean')    
        #ax.plot(index_pred[cutoff:],mean_old*np.ones((len(index_pred[cutoff:]))),col[1],label = 'old mean')

        ax.fill_between(index_pred[-n_pred:],lower_out,upper_out,color="k",alpha=0.25)

        ax.xaxis.set_tick_params(rotation=20, labelsize=10)
        ax.set_xlabel('Time')
        if perc == 1:
            ax.set_ylabel('Market Shares')
        else:
            ax.set_ylabel('Gram')
        plt.title(TITLE)
        #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
        #      ncol=2, fancybox=True, shadow=True)
        fig.savefig(path.join(outpath,"%s.png"%dat.name),dpi=300)
        plt.close()
        fig.clear()
            
    print("done")
    return outpath + "\\"   


# Prints png of the past 12 months with a time-series analysis of the past 6 months and a prediction. Requires dataframe with 1 column
# add = 'Raw' (default), adds a string to the end of the name of the prints in the folder
# out_path add = '' (default), creates a subfolder to store the prints if it is not empty
# conv = 1 (default), increase in case the values of the data is very small (e.g. 0.001))
# TITLE = '' (default), adds a title to the graph
def Brick_edge_png_single(df,cut=-12,n_pred = 1,outpath_add='',conv=1,TITLE='',perc=0):
    if len(outpath_add ) == 0:
        outpath_add = "Edge"
        
    if os.path.exists(outpath_plot + outpath_add) != True:
        os.makedirs(os.path.join("Plots", outpath_add))
        print("Adding new folder: %s"%outpath_add)
        outpath = outpath_plot + outpath_add
    else:
        outpath = outpath_plot + outpath_add

    name = df.columns[0]
    
    col = ['g--','r--']
    
#    perc = 0
#    if (df.iloc[0,0]) < 1:
#        perc = 1

#    dfi=df    
#    imp = SimpleImputer(missing_values=np.nan,strategy='mean')
#    dfi=pd.DataFrame(imp.fit_transform(dfi))
#    dfi.columns=df.columns
#    dat = dfi.copy()*conv
    dat = df.copy()*conv
    result = arimamodel(dat)
    dat = dat/conv
    mean_old = dat[-7:-1].mean()
    mean_new = dat[-6:].mean()

    n = len(dat.index)
    n_tot = n + n_pred
  
    prediction, prediction_conf_int = result.predict(n_periods = n_pred,return_conf_int=True)
    prediction, prediction_conf_int = prediction/conv, prediction_conf_int/conv
    in_sample, in_sample_conf_int = result.predict_in_sample(return_conf_int=True)
    in_sample, in_sample_conf_int = in_sample/conv, in_sample_conf_int/conv

    lower_in = in_sample_conf_int[:,0]
    upper_in = in_sample_conf_int[:,1]
    lower_out = prediction_conf_int[:,0]
    upper_out = prediction_conf_int[:,1]

    index_pred = TimePeriodPrediction(dat.index,start_date,n_pred=n_pred).strftime('%Y-%m')
    cutoff = cut - n_pred

    in_pred = pd.Series(np.full((n_tot),np.nan))
    in_pred[:n] = in_sample
    dat_pred = pd.Series(np.full((n_tot),np.nan))
    dat_pred[:n],dat_pred[n:] = dat.iloc[:,0],prediction
    dat1 = pd.Series(np.full((n_tot),np.nan))
    dat1[:n] = dat.iloc[:,0]
    pred1 = pd.Series(np.full((n_tot),np.nan))
    pred1[n-1:n],pred1[n:] = dat.iloc[-1],prediction
    pred2 = pd.Series(np.full((n_tot),np.nan))
    pred2[:n],pred2[n:] = in_sample,prediction.copy()
    lower = pd.Series(np.zeros((n_tot)))
    lower[:n],lower[n:] = lower_in,lower_out
    upper = pd.Series(np.zeros((n_tot)))
    upper[:n],upper[n:] = upper_in,upper_out

    col = ['g--','r--']

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(index_pred[cutoff:],pred2[cutoff:],'r')
    ax.plot(index_pred[cutoff:],dat_pred[cutoff:],'k')
    ax.plot(index_pred[cutoff:],pred1[cutoff:],'g')
    ax.plot(index_pred[cutoff:],dat1[cutoff:],'k',label = name)
    ax.plot(index_pred[cutoff:],lower[cutoff:],'b--',label = 'CI')
    ax.plot(index_pred[cutoff:],upper[cutoff:],'b--')

    #ax.plot(index_pred[cutoff:],mean_new[0]*np.ones((len(index_pred[cutoff:]))),col[0],label = 'new mean')    
    #ax.plot(index_pred[cutoff:],mean_old[0]*np.ones((len(index_pred[cutoff:]))),col[1],label = 'old mean')

    ax.fill_between(index_pred[-n_pred:],lower_out,upper_out,color="k",alpha=0.25)
    
    ax.xaxis.set_tick_params(rotation=20, labelsize=10)
    ax.set_xlabel('Time')
    if perc == 1:
        ax.set_ylabel('Market Shares')
    else:
        ax.set_ylabel('Gram')
    plt.title(TITLE)
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
    #          ncol=2, fancybox=True, shadow=True)
    fig.savefig(path.join(outpath,"%s.png"%name),dpi=300)
    plt.close()
    fig.clear()
            
    print("done")
    return outpath + "\\"


# Combines 2 png files. Used for combining 2 graphs
# df, the dataframe. The combinations will be based on the columns of this dataframe
# outpath1, the outpath for the first graph. Make sure the the names of the graphs in this path are the same as that of the columns of the dataframe
# outpath2, the outpath for the second graph. Make sure the the names of the graphs in this path are the same as that of the columns of the dataframe
# name1_add, name2_add. Use this if the names of the graphs have additions in their names
# outpath_add = '', change the folder name and outpath of the combined graphs
def Comb_png(df,outpath1,outpath2,name1_add='',name2_add='',outpath_add=''):
    if len(outpath_add ) == 0:
        outpath_add = "Comb"
        
    if os.path.exists(outpath_plot + outpath_add) != True:
        os.makedirs(os.path.join("Plots", outpath_add))
        print("Adding new folder: %s"%outpath_add)
        outpath = outpath_plot + outpath_add
    else:
        outpath = outpath_plot + outpath_add
    
    
    name = df.columns 
    for i in range(len(df.columns)):
        
        if os.path.exists("%s%s.png"%(outpath1,name[i])) != True:
            logo1 = r"%s%s .png"%(outpath1,name[i])
        else:
            logo1 = r"%s%s.png"%(outpath1,name[i])
        if os.path.exists("%s%s.png"%(outpath2,name[i])) != True:
            logo2 = r"%s%s .png"%(outpath2,name[i])
        else:
            logo2 = r"%s%s.png"%(outpath2,name[i])
            
        fig,ax = plt.subplots(nrows = 1, ncols = 2, figsize=(12,6))
            
        plt.subplot(1,2,1)
        img=plt.imread(logo1)
        imgplot = plt.imshow(img)
        plt.axis('off')
    
        plt.subplot(1,2,2)
        img=plt.imread(logo2)
        imgplot = plt.imshow(img)
        plt.axis('off')
        plt.tight_layout(True)
        plt.close()

        fig.savefig(path.join(outpath,"%s.png"%name[i]), bbox_inches='tight',dpi=200)
        fig.clear()
        
    print("done")
    return outpath + "\\"

# Calculates Autoregressive models for every time-series in the dataframe
def Over(df):
    sub=[]
    dfi=[]
    coefficients = pd.DataFrame(np.zeros((len(df.columns),4)))
    coefficients.columns = ['Const','Lag 1','Lag 2','Sigma2']
    prediction = df.copy()*0
    over_bound = prediction[-1:]*0
    over_bound_brick = []
    sub=pd.DataFrame(df.isna().sum(axis=0))
    sub.columns=['count_na']
    dfi=df
    n_df = len(df.index)
    if range(len(df.columns != 1 )):
        for i in range(len(df.columns)):
            if sub.iloc[i,-1] == n_df:
                dfi.iloc[0,i]=0

    for i in range(len(df.columns)):
        imp = SimpleImputer(missing_values=np.nan,strategy='mean')
        dfi=pd.DataFrame(imp.fit_transform(dfi))
        dfi.columns=df.columns
        result = arimamodel(dfi.iloc[:,i])  # Fits each time-series to the model
        in_sample, in_sample_conf_int = result.predict_in_sample(return_conf_int=True)  # Returns the in sample predictions and confidence intervals
        prediction.iloc[:,i] = in_sample
        if df.iloc[-1,i] < in_sample_conf_int[-1,0] or df.iloc[-1,i] > in_sample_conf_int[-1,1]:  # Returns 1 if thee confidence intervals have been exceeded
            over_bound.iloc[:,i] = 1
            over_bound_brick.append(df.columns[i])
        coefficients.iloc[i,0:result.order[0]+1] = result.params()[0:result.order[0]+1] # Fills the matrix with coefficients from the fitted time-series models
        coefficients.iloc[i,-1] = result.params()[-1]
    coefficients.index=df.columns
    coefficients = coefficients.loc[:, (coefficients != 0).any(axis=0)]
    return coefficients,prediction,over_bound,over_bound_brick


# Returns a dataframe with 1 if the confidence interval has been exceeded at any point in time and 0 if not
# Doesn't do anthing in this program, but can be useful when you fit the time-series models manually
def CI_over(df,CI,single=0):
    over = np.abs(df[1:].copy())*0

    if single != 0:
        for i in range(len(over.index)):
            if df.iloc[i] <= CI.iloc[i,0]:
                over.iloc[i] = 1
            elif df.iloc[i] >= CI.iloc[i,1]:
                over.iloc[i] = 1
    if single == 0:
        for j in range(len(df.columns)):
            for i in range(len(over.index)):
                if df.iloc[i+1,j] <= CI[j].iloc[i,0]:
                    over.iloc[i,j] = 1
                elif df.iloc[i+1,j] >= CI[j].iloc[i,1]:
                    over.iloc[i,j] = 1
    return over

# Does an Augmented Dickey-Fuller test on the dataframe to tell what which time-series are stationary
def Stationarity(df):
    AD = pd.DataFrame(np.zeros((3,len(df.columns))))
    AD.columns = df.columns
    AD.index = ['t-stat','p-value','Stationary']
    for i in range(len(df.columns)):
        AD.iloc[0:2,i] = adfuller(df.iloc[:,i])[0:2]
        if adfuller(df.iloc[:,i])[1] <= 0.05:
            AD.iloc[2,i] = 1     
    return AD.T 



def model(timeseries):
    y_ = df.iloc[:,i][1:].copy()
    x_ = pd.DataFrame(df_inc.iloc[:,i].copy()) * 0 + 1
    x_['1'] = df_inc.iloc[:,i].copy()
    b_ = np.dot(inv(np.dot(x_.T,x_)),np.dot(x_.T,y_))
    y_h = np.dot(x_,b_)
    res_ = y_-y_h
    sd_res_ = np.sqrt(np.var(res_))
    c_l, c_u = y_h-1.96*sd_res_,y_h+1.96*sd_res_


# Fits the data to an automated Autoregressive model
# NOTE!!!!: Does not work on time-series containing missing values!!!
def arimamodel(timeseries):
    automodel = pm.auto_arima(timeseries, 
                              start_p=1,
                              d=1,
                              max_p=1,
                              start_q=0,
                              max_q=0,
                              test="adf",
                              #test='kpss',
                              disp=1,
                              seasonal=False,
                              trace=False,
                              alpha=0.99,
                              m=1,
                              stepwise=True,
                              information_criterion='aic',
                              suppress_warnings=True)
    return automodel



def Complete_subset(data,df_main,product='',brick='BRK66',target='Gram',prd='PRD_PLUS'):
    DF = df_main.copy()*0

    for i in range(len(DF.columns)):
        if len(product) > 0:
            data_sub = data[data[prd]==product]
        else:
            data_sub = data
        data_sub = pd.DataFrame(data_sub[data_sub[brick]==df_main.columns[i]].groupby('TIME_PERIOD').sum()[target])
        data_sub_comp = Subset(data_sub,df_main)
        DF.iloc[:,i] = data_sub_comp
    DF0 = DF.copy()
    DF0[np.isnan(DF0)] = 0
    return DF0,DF

def Subset(df,df_main):
    prod = df_main.iloc[:,0].copy()*np.nan
    prod_index = prod.index
    prd = df.copy()
    prd_index = prd.index
    m = 0
    for i in range(len(prod_index)):
        if m < len(prd_index):
            if  prod_index[i].split('-')[0] == prd_index[m].split(' ')[1][:-2] and  prod_index[i].split('-')[1] ==  prd_index[m].split(' ')[1][-2:]:
                prod.iloc[i] = prd.iloc[m].values
                m = m+1
    return prod

def ZeroToNan(df0):
    df = df0.copy()
    df[df==0]=np.nan
    return df

def perc(df): return '{percent:.2%}'.format(percent=df)

def Quarterly(df):
    dfQ = df.groupby(pd.PeriodIndex(df.index, freq='Q'), axis=0).sum()
    dfQ.index = df.index.strftime('%YQ%q')
    return dfQ


registerFont(TTFont('arial', 'arial.ttf'))
registerFont(TTFont('arial-bold', 'arialbd.ttf'), )
registerFont(TTFont('arial-italic', 'ariali.ttf'), )
registerFont(TTFont('arial-bolditalic', 'arialbi.ttf'), )


os.chdir(r"C:\Users\nsharma\Documents\Work - Neha Sharma\S&T\Netherlands\02. Adhoc Projects\2019 Projects\Astellas Early Warning System\202001")
#os.chdir(r"C:\Users\ksandeep\Documents\Astellas Early Warning System (Dylan)\EarlyWarningIndicator\Astellas")
cwd = os.getcwd() + '\\'
cwd

#import Functions_AEWS_v1
data = pd.read_csv( cwd + "A308_projected_202001.csv")       # Read the csv data
factor = pd.read_excel(cwd+"Advagraf_Factor_Checks.xlsx")    # read the Factor data, used to convert CU_projected to Grams

outpath_add_plot = "Plots"
if os.path.exists(cwd + outpath_add_plot) != True:
    os.makedirs(outpath_add_plot)
    print("Adding new folder: %s"%outpath_add_plot)
    outpath_plot = cwd + outpath_add_plot + '\\'
else:
    outpath_plot = cwd + outpath_add_plot + '\\'

data = data[data['BRK66'] != '99 Rest gebied'] # Exludes this brick from the dataset
data_with_factor = pd.merge(data,factor,on='PFC',how='left')
data = data_with_factor
del data_with_factor
data['Gram'] = (data['CU_projected']*data['Factor'])/1000 # Translates milligrams to grams
data.to_excel('check_grams_PRD.xlsx')
data = data[data['ATC3'] == 'L04X0 OVERIGE IMMUNOSUPPRESIVA'] # Filter for only L04X ATC
data.ATC3.unique()

data = data[data['MOLECULE'] == 'TACROLIMUS'] # Filter for only Tacrolimus molecule
data.MOLECULE.unique()

#check = pd.crosstab(data.MOLECULE, data.PRD_PLUS).apply(lambda r: r/r.sum(), axis = 1)

BRK_names = data['BRK66'].unique()     # Categorize the Bricks  
#MNF_names = data['MNF_PLUS'].unique()   # Categorize the manufacturers
#MOL_names = data['MOLECULE'].unique()   # Categorize the molecules
PRD_names = data['PRD_PLUS'].unique()

#Recode TACROLIMUS-A6D ,TACROLIMUS-SDZ, TACROLIMUS-CTP to TACROLIMUS-GEN
data.loc[(data.PRD_PLUS == 'TACROLIMUS-A6D'),'PRD_PLUS']='TACROLIMUS-GEN'
data.loc[(data.PRD_PLUS == 'TACROLIMUS-SDZ'),'PRD_PLUS']='TACROLIMUS-GEN'
data.loc[(data.PRD_PLUS == 'TACROLIMUS-CTP'),'PRD_PLUS']='TACROLIMUS-GEN'
data.PRD_PLUS.unique()

#Adding the suffix of OD and TD in the product names
data.loc[(data.PRD_PLUS == 'ADPORT'),'PRD_PLUS']='ADPORT_TD'
data.loc[(data.PRD_PLUS == 'ADVAGRAF'),'PRD_PLUS']='ADVAGRAF_OD'
data.loc[(data.PRD_PLUS == 'ENVARSUS'),'PRD_PLUS']='ENVARSUS_TD'
data.loc[(data.PRD_PLUS == 'PROGRAFT'),'PRD_PLUS']='PROGRAFT_TD'
data.loc[(data.PRD_PLUS == 'MODIGRAF'),'PRD_PLUS']='MODIGRAF_TD'
data.loc[(data.PRD_PLUS == 'DAILIPORT-SDZ'),'PRD_PLUS']='DAILIPORT-SDZ_OD'
data.loc[(data.PRD_PLUS == 'TACNI'),'PRD_PLUS']='TACNI_TD'
data.loc[(data.PRD_PLUS == 'TACROLIMUS-GEN'),'PRD_PLUS']='TACROLIMUS-GEN_TD'
data.PRD_PLUS.unique()


TIME = sorted(data['TIME_PERIOD'].unique())      # Sort the time
T = len(TIME)

start_date = TIME[0]
end_date = TIME[-1]

brick = 'BRK66'
MOLECULES = 'MOLECULE'
MOL = 'TACROLIMUS'
PRD_PLUS ='PRD_PLUS'
Bricks = data.BRK66.unique()
PRD = ['ADPORT_TD', 'ADVAGRAF_OD', 'ENVARSUS_TD', 'PROGRAFT_TD',
       'MODIGRAF_TD', 'TACROLIMUS-GEN_TD', 'DAILIPORT-SDZ_OD', 'TACNI_TD']
PROD_TYPE = 'PROD_TYPE'
target = 'Gram'
data_prd_single = {}
data_prd_rest = {}
data_prd_all = {}
df = {}
df_all = {}
df_rest = {}
df_diff = {}
df_inc = {}
df_rest_diff = {}
df_rest_inc = {}
df_diff_all = {}
df_all_inc = {}
import numpy
#data.to_excel('data_check.xlsx')
for i in range(len(PRD)):
    data_prd_single[i],data_prd_rest[i],data_prd_all[i] = Data_sub_cu(data,brick,target,sub2=PRD_PLUS,sub2_name=PRD[i]) #subset the data
    # data_prd_single includes only product's data 
    # data_prd_rest includes rest all products in Tacrolimus molecule
    # data_prd_all inludes all products data at National level 
    #----------------------------------------------------
    # Normal dataframe
    df[i], df_all[i], df_rest[i] = data_prd_single[i].copy(), data_prd_all[i].copy(), data_prd_rest[i].copy()
    df_diff[i], df_inc[i] = First_difference(df[i]) # A dataframe with the first differences of df and its relative increase 
    df_rest_diff[i], df_rest_inc[i] = First_difference(df_rest[i])  # A dataframe with the first differences of df_rest and its relative increase 
    df_diff_all[i], df_all_inc[i]= First_difference(numpy.array(df_all[i]),single=1)  # A dataframe with the first differences of df_all and its relative increase 
    df_diff_all[i] = pd.DataFrame(df_diff_all[i]) 
    df_diff_all[i].index = df_diff[i].index
    df_diff_all[i].columns = ['Nederland']
    df_all_inc[i] = pd.DataFrame(df_all_inc[i])
    df_all_inc[i].index = df_diff[i].index
    df_all_inc[i].columns = ['Nederland']

df_all_mean_old ={}
df_all_mean_new={}
df_all_mean_inc={}
df_all_mean_inc_perc={}
df_shares={}
df_rest_shares={}
df_all_shares={}
df_all_rest_shares={}
df_all_shares_diff={}
df_all_shares_inc={}
for i in range(len(PRD)):
    df_all_mean_old[i] = df_all[i].iloc[-7:-1].mean()
    df_all_mean_new[i] = df_all[i].iloc[-6:].mean()
    df_all_mean_inc[i] = df_all_mean_new[i] - df_all_mean_old[i]   # Calculate increases in mean
    df_all_mean_inc_perc[i] = df_all_mean_inc[i]/df_all_mean_old[i]  # Calculate relative increases in mean
    # Normal shares dataframe
    df_shares[i] = df[i]/(df[i]+df_rest[i])
    df_rest_shares[i] = 1-df_shares[i]
    df_all_shares[i] = pd.DataFrame(np.sum(df[i], axis=1) / (np.sum(df[i], axis=1) + np.sum(df_rest[i], axis=1)))
    df_all_shares[i].columns = ['Nederland']
    df_all_rest_shares[i] = 1 - df_all_shares[i]
    df_all_shares_diff[i], df_all_shares_inc[i] = First_difference(pd.DataFrame(df_all_shares[i]))

#---------------------------------------------------------------------
# Rolling window and shares dataframe
df_rolling={}
df_rest_rolling={}
df_all_rolling={}
df_rolling_diff={}
df_rolling_inc={}
df_rest_rolling_diff={}
df_rest_rolling_inc={}
df_rolling_shares={}
df_rolling_shares_diff={}
df_rolling_shares_inc={}
df_rest_rolling_shares={}
df_all_rolling_shares={}
for i in range(len(PRD)):
    df_rolling[i],df_rest_rolling[i],df_all_rolling[i] = df[i].rolling(window=3).sum()[2:], df_rest[i].rolling(window=3).sum()[2:],df_all[i].rolling(window=3).sum()[2:]
    # df_rolling includes only product's data with rolling window data over 3 months
    # df_rest includes rest all products in Tacrolimus molecule with rolling window data over 3 months
    # df_all inludes only product's data with rolling window data over 3 months from Astellas     
    df_rolling_diff[i], df_rolling_inc[i] = First_difference(df_rolling[i])  # A dataframe with the first differences of df_rolling and its relative increase 
    df_rest_rolling_diff[i], df_rest_rolling_inc[i] = First_difference(df_rest_rolling[i]) # A dataframe with the first differences of df_rolling_rest and its relative increase 
    df_rolling_shares[i] = df_rolling[i]/(df_rest_rolling[i]+df_rolling[i]) # A dataframe of the rolling window market shares data 
    df_rolling_shares_diff[i],df_rolling_shares_inc[i] = First_difference(df_rolling_shares[i]) # A dataframe with the first differences of df_rolling_shares and its relative increase 
    df_rest_rolling_shares[i] = 1 - df_rolling_shares[i]
    df_all_rolling_shares[i] = pd.DataFrame(df_rolling[i].sum(axis=1)/(df_rest_rolling[i].sum(axis=1)+df_rolling[i].sum(axis=1)))  # Calculate national rolling window market share data
    df_all_rolling_shares[i].columns = ['Nederland']




#----------------------------------------------------------------------
# Quarterly and shares dataframe  (Currently this non of the quarterly results are in use)
#df_quarterly={}
#df_all_quarterly={}
#df_rest_quarterly={}
#df_quarterly_diff={}
#df_quarterly_inc={}
#df_rest_quarterly_diff={}
#df_rest_quarterly_inc={}
#df_all_quarterly_diff={}
#df_all_quarterly_inc={}
#df_quarterly_shares={}
#df_quarterly_rest_shares={}
#df_all_quarterly_shares_diff={}
#df_all_quarterly_shares_inc={}
#for i in range(len(PRD)):  
#    df_quarterly[i] = Quarterly(df[i])
#    df_all_quarterly[i] = Quarterly(df_all[i])
#    df_rest_quarterly[i] = Quarterly(df_rest[i])
#    df_quarterly_diff[i], df_quarterly_inc[i] = First_difference(pd.DataFrame(df_quarterly[i]))
#    df_rest_quarterly_diff[i], df_rest_quarterly_inc[i] = First_difference(pd.DataFrame(df_rest_quarterly[i]))
#    df_all_quarterly_diff[i], df_all_quarterly_inc[i] = First_difference(pd.DataFrame(df_all_quarterly[i]))
#    #outpath_raw_quarterly = Brick_to_png_raw(df_all_quarterly,outpath_add='Raw quarterly')
#    #outpath_edge_quarterly = Brick_edge_png_single(df_all_quarterly,outpath_add='Edge quarterly')    
#    df_quarterly_shares[i] = df_quarterly[i]/(df_quarterly[i]+df_rest_quarterly[i])
#    df_quarterly_rest_shares[i] = 1-df_quarterly_shares[i]
#    df_all_quarterly_shares[i] = pd.DataFrame(np.sum(df_quarterly[i], axis=1) / (np.sum(df_quarterly[i], axis=1) + np.sum(df_rest_quarterly, axis=1)))
#    df_all_quarterly_shares[i].columns = ['Nederland']
#    df_all_quarterly_rest_shares[i] = 1 - df_all_quarterly_shares[i]    
#    df_all_quarterly_shares_diff[i], df_all_quarterly_shares_inc[i] = First_difference(pd.DataFrame(df_all_quarterly_shares[i]))


#---------------------------------------------------------------------
# Ranks of the bricks
#Getting brick level data for all products of molecule TACROLIMUS
df_tac = pd.pivot_table(data,index='TIME_PERIOD',columns='BRK66',values=target,aggfunc='sum')
df_tac_all = pd.DataFrame(data.groupby('TIME_PERIOD').sum()[target])
df_tac_all.rename(columns={target: 'Nederland'},inplace=True)

df_tac_diff, df_tac_inc = First_difference(df_tac) # A dataframe with the first differences of df and its relative increase 
df_tac_all_diff, df_tac_all_inc= First_difference(numpy.array(df_tac_all),single=1)  # A dataframe with the first differences of df_all and its relative increase 

df_tac_all_diff = pd.DataFrame(df_tac_all_diff) 
df_tac_all_diff.index = df_tac_diff.index
df_tac_all_diff.columns = ['Nederland']
df_tac_all_inc = pd.DataFrame(df_tac_all_inc)
df_tac_all_inc.index = df_tac_diff.index
df_tac_all_inc.columns = ['Nederland']

#Needs verification - Sandeep K
df_tac_all_mean_old = df_tac_all.iloc[-7:-1].mean()
df_tac_all_mean_new = df_tac_all.iloc[-6:].mean()
df_tac_all_mean_inc = df_tac_all_mean_new - df_tac_all_mean_old  # Calculate increases in mean
df_tac_all_mean_inc_perc = df_tac_all_mean_inc / df_tac_all_mean_old  # Calculate relative increases in mean

data_tac = data[data[MOLECULES] == MOL]

data_gen = data_tac[data_tac["PRD_PLUS"].isin(['ADVAGRAF_OD','PROGRAFT_TD']) == False]
data_spec = data_tac[data_tac["PRD_PLUS"].isin(['ADVAGRAF_OD','PROGRAFT_TD']) == True]
df_gen=pd.pivot_table(data_gen,index='TIME_PERIOD',columns='BRK66',values=target,aggfunc='sum')
df_spec=pd.pivot_table(data_spec,index='TIME_PERIOD',columns='BRK66',values=target,aggfunc='sum')
df_gen0 = df_gen.fillna(0)
df_spec0 = df_spec.fillna(0)

df_gen_tot = pd.DataFrame(df_gen0.T.sum()) # National level of generiek
df_gen_tot.columns = ['Generiek']
df_spec_tot = pd.DataFrame(df_spec0.T.sum())  # National level of specialite
df_spec_tot.columns = ['Specialite']

#df_gen0,df_gen = Complete_subset(data_gen,df_tac) # A dataframe containing only generiek data, in df_gen0 the NA values have been replaced with 0
#df_spec0,df_spec = Complete_subset(data_spec,df_tac)  # A dataframe containing only specialite data, in df_spec0 the NA values have been replaced with 0
#df_gen0.to_excel('df_gen0.xlsx')
#df_gen.to_excel('df_gen.xlsx')

top, cutoff = 10 , -6    # top: number of bricks that are at the top of the month, cutoff: number of past months to be used calculating the means
crit_neg = [-0.2,-0.4,-0.6,-0.8,-1]  
crit_pos = [0.2,0.4,0.6,0.8,1]
crit_over, crit_count = Count(df_tac_inc,crit_neg,crit_pos,cutoff=cutoff) # Returns a dataframe showing when the data has exceeded the values from crit_neg and crit_pos and how many (count)
crit_over_all, crit_count_all = Count(df_tac_all_inc,crit_neg,crit_pos,cutoff=cutoff)  # Returns a dataframe showing when the data has exceeded the values from crit_neg and crit_pos and how many (count)

TOP_old, TOP_new, TOP_old_diff, TOP_new_diff = Top(df_tac,cutoff)  
rank, rank_top, rank_rest, rank_left = Rank(df_tac,top,TOP_old,TOP_new)
# TOP_old: bricks belonging to the old top, TOP_new: bricks belonging to the new top
# TOP_old_diff: bricks that left the top rankings, TOP_new_diff: bricks that made the new top rankings
# rank: list of rankings, rank_top: list of top rankings, rank_left, list of non-top rankings

dist,dist_round = Dist(df_tac,rank['new'])   # Returns some simple analysis on the dataframe and a rounded version

dist_shares={}
dist_shares_round={}
dist_all={}
dist_all_round={}
dist_all_shares = {}
dist_all_shares={}
dist_all_shares_round={}
dist_rolling={}
dist_rolling_round={}
dist_all_rolling={}
dist_all_rolling_round={}
dist_rolling_shares={}
dist_rolling_shares_round={}
dist_all_rolling_shares={}
dist_all_rolling_shares_round={}
for i in range(len(PRD)):  
    dist_shares[i],dist_shares_round[i] = Dist(df_shares[i],rank['new'])
#    dist_quarterly,dist_quarterly_round = Dist(df_quarterly,rank['new'])
#    dist_quarterly_shares,dist_quarterly_shares_round = Dist(df_quarterly_shares,rank['new'])
    dist_all[i],dist_all_round[i] = Dist(df_all[i],rank['new'])
    dist_all_shares[i],dist_all_shares_round[i] = Dist(df_all_shares[i],rank['new'])
#    dist_all_quarterly,dist_all_quarterly_round = Dist(df_all_quarterly,rank['new'])
#    dist_all_quarterly_shares,dist_all_quarterly_shares_round = Dist(df_all_quarterly_shares,rank['new'])
    dist_rolling[i],dist_rolling_round[i] = Dist(df_rolling[i],rank['new'])
    dist_all_rolling[i],dist_all_rolling_round[i] = Dist(df_all_rolling[i],rank['new'])
    dist_rolling_shares[i],dist_rolling_shares_round[i] = Dist(df_rolling_shares[i],rank['new'])
    dist_all_rolling_shares[i],dist_all_rolling_shares_round[i] = Dist(df_all_rolling_shares[i],rank['new'])



#--------------------------------------------------------------------------------------
# Creates a table with the ranking including AES
# Add the brick numbers that make the top 10 according to the client in brick_AES
brick_AES = ['02','20','25','33','35','38','46','65']
rank_AES = rank[rank['brick'].isin(brick_AES)]
rank_AES_top = rank_AES[rank_AES['new'] <= 10]
rank_AES_rest = rank_AES[rank_AES['new'] > 10]
brick_same_AES = (set(rank['brick'][:10].values)).union(set(brick_AES))
rank_same_AES = rank[rank['brick'].isin(brick_same_AES)]
AES = ['no' for x in range(len(rank_same_AES.index))]
for i in range(len(rank_same_AES.index)):
    for j in range(len(brick_AES)):
        if rank_same_AES['brick'][i] == brick_AES[j]:
            AES[i] = 'yes'
rank_same_AES.insert(3,'AES',AES)  # A top 10 list including the top 10 from the client
rank_diff_AES = rank[~rank.index.isin(rank_same_AES.index)]  # The opposite of rank_same_AES


#---------------------------------------------------------------
# Create table png of rank AES
outpath_plot_tables = outpath_plot + "Tables\\"
if os.path.exists(outpath_plot_tables) == False:
    os.makedirs(os.path.join("Plots", "Tables"))

fig, ax = plt.subplots(figsize=(5,5))

labels = ['Brick','New Top','Old Top','AES']
ax.axis('off')
rank_table = ax.table(cellText=rank_same_AES.values,cellLoc='center', colLabels=labels, bbox=[0,0,1,1])
for i in range(4):
    rank_table[0,i].set_facecolor("#00A3E0") 
    rank_table[0,i].get_text().set_color('white')
fig.savefig(path.join(outpath_plot_tables,"Rank AES.png"),dpi=300)
plt.close()


#------------------------------------------------------------------------
# Fit the time-sries within the dataframe to the autoregressive models
# df_beta: the coefficents of the models
# df_prediction: the in-sample predictions of the models
# df_over_bound_brick: The bricks that have exceeded the confidence intervals at the current month
df_beta={}
df_prediction={}
df_over_bound={}
df_over_bound_brickp={}
df_all_beta={}
df_all_prediction={}
df_all_over_bound={}
df_all_over_bound_brick={}
df_rolling_shares_beta={}
df_rolling_shares_prediction={}
df_rolling_shares_over_bound={}
df_rolling_shares_over_bound_brick={}
#df_all_rolling_shares_beta={}
#df_all_rolling_shares_prediction={}
#df_all_rolling_shares_over_bound={}
#df_all_rolling_shares_over_bound_brick={}
#ADF={}
for i in range(len(PRD)):
    df_all_beta[i],df_all_prediction[i],df_all_over_bound[i],df_all_over_bound_brick[i] = Over(df_all[i])
    df_beta[i],df_prediction[i],df_over_bound[i],df_over_bound_brickp[i] = Over(df[i])
# Code below still in progress - ARIMA model errors.
#for i in range(len(PRD)):
#    df_rolling_shares_beta[i],df_rolling_shares_prediction[i],df_rolling_shares_over_bound[i],df_rolling_shares_over_bound_brick[i] = Over(df_rolling_shares[i])

#    df_all_rolling_shares_beta[i],df_all_rolling_shares_prediction[i],df_all_rolling_shares_over_bound[i],df_all_rolling_shares_over_bound_brick[i] = Over(df_all_rolling_shares[i])
#    ADF = Stationarity(df[0]) # Gives the statistical test values and Stationary = 1 if stationary, 0 otherwise

#Need verification - Sandeep K
df_tac_beta,df_tac_prediction,df_tac_over_bound,df_tac_over_bound_brick = Over(df_tac)
df_tac_all_beta,df_tac_all_prediction,df_tac_all_over_bound,df_tac_all_over_bound_brick = Over(df_tac_all)
#df_tac_rolling_shares_beta,df_tac_rolling_shares_prediction,df_tac_rolling_shares_over_bound,df_tac_rolling_shares_over_bound_brick = Over(df_tac_rolling_shares)
#df_tac_all_rolling_shares_beta,df_tac_all_rolling_shares_prediction,df_tac_all_rolling_shares_over_bound,df_tac_all_rolling_shares_over_bound_brick = Over(df_tac_all_rolling_shares)

#------------------------------------------------------------------------
# ..._raw, create plots of the data
# ..._edge or ..._edge_..., analyse the data, create time-series models, plot the last 12 months of the data and some analysis including a one-month prediction
# They are all named outpath because the function returns the path to the saved images
conv=1000000  # Handy when dealing with data that contains only small values
outpath_raw_national = Brick_to_png_raw(df_tac_all,single=1,TITLE='')    # Added by Sandeep K
outpath_edge_national = Brick_edge_png_single(df_tac_all,TITLE='')    # Added by Sandeep K
#outpath_edge_rolling_shares_national = Brick_edge_png_single(df_tac_all_rolling_shares,outpath_add='Edge Rolling Shares - National',conv=conv,TITLE='')    # Uncomment after definiton of 'df_tac_all_rolling_shares'
#outpath_comb_national = Comb_png(df_tac_all,outpath_edge_national,outpath_edge_rolling_shares_national)    # Uncomment after definition of 'df_tac_all_rolling_shares'
outpath_raw={}
raw_PRD = ['raw_ADPORT_TD', 'raw_ADVAGRAF_OD', 'raw_ENVARSUS_TD', 'raw_PROGRAFT_TD',
       'raw_MODIGRAF_TD', 'raw_TACROLIMUS-GEN_TD', 'raw_DAILIPORT-SDZ_OD', 'raw_TACNI_TD']
for i in range(len(raw_PRD)):
    outpath_raw[i] = Brick_to_png_raw(df[i],TITLE='Volume Development',outpath_add=raw_PRD[i])   
    outpath_raw[i] = Brick_to_png_raw(df_all[i],single=1,TITLE='Volume Development',outpath_add=raw_PRD[i]) 

outpath_raw[len(raw_PRD)] = Brick_to_png_raw(df_gen_tot,single=1,TITLE='Generiek-Volume Development')  
outpath_raw[len(raw_PRD)+1] = Brick_to_png_raw(df_spec_tot,single=1,TITLE='Specialite-') 

edge_PRD = ['edge_ADPORT_TD', 'edge_ADVAGRAF_OD', 'edge_ENVARSUS_TD', 'edge_PROGRAFT_TD',
       'edge_MODIGRAF_TD', 'edge_TACROLIMUS-GEN_TD', 'edge_DAILIPORT-SDZ_OD', 'edge_TACNI_TD']
outpath_edge={}
for i in range(len(edge_PRD)):
   outpath_edge[i] = Brick_edge_png_single(df_all[i],TITLE='Volume Development with Prediction',outpath_add=edge_PRD[i],perc=0)
   outpath_edge[i] = Brick_edge_png(df[i],TITLE='Volume Development with Prediction',outpath_add=edge_PRD[i],perc=0)
 
outpath_edge[len(edge_PRD)] = Brick_edge_png_single(df_gen_tot,TITLE='Generiek-Volume Development with Prediction',perc=0) 
outpath_edge[len(edge_PRD)+1] = Brick_edge_png_single(df_spec_tot,TITLE='Specialite-Volume Development with Prediction',perc=0)

edge_RS_PRD = ['edge_RS_ADPORT_TD', 'edge_RS_ADVAGRAF_OD', 'edge_RS_ENVARSUS_TD', 'edge_RS_PROGRAFT_TD',
       'edge_RS_MODIGRAF_TD', 'edge_RS_TACROLIMUS-GEN_TD', 'edge_RS_DAILIPORT-SDZ_OD', 'edge_RS_TACNI_TD']

outpath_edge_rolling_shares={}
for i in range(len(edge_RS_PRD)):
    outpath_edge_rolling_shares[i] = Brick_edge_png_single(df_all_rolling_shares[i],outpath_add=edge_RS_PRD[i],conv=conv,TITLE='Rolling Market Share',perc=1)
    outpath_edge_rolling_shares[i] = Brick_edge_png(df_rolling_shares[i],outpath_add=edge_RS_PRD[i],conv=conv,TITLE='Rolling Market Share',perc=1)

comb_PRD = ['comb_ADPORT_TD', 'comb_ADVAGRAF_OD', 'comb_ENVARSUS_TD', 'comb_PROGRAFT_TD',
       'comb_MODIGRAF_TD', 'comb_TACROLIMUS-GEN_TD', 'comb_DAILIPORT-SDZ_OD', 'comb_TACNI_TD']
# Combines two graphs given their paths
outpath_comb={}
for i in range(len(comb_PRD)):
    outpath_comb[i] = Comb_png(df[i],outpath_edge[i],outpath_edge_rolling_shares[i],outpath_add=comb_PRD[i]) 
    outpath_comb[i] = Comb_png(df_all[i],outpath_edge[i],outpath_edge_rolling_shares[i],outpath_add=comb_PRD[i]) 


#---------------------------------------------------------------------
# Create the pdf

class MyDocTemplate(BaseDocTemplate):  # Create a custom template for the pdf
    def __init__(self, filename, **kw):
        self.allowSplitting = 0
        BaseDocTemplate.__init__(self, filename, **kw)
        template = PageTemplate('normal', [Frame(1.7*cm, 1.7*cm, 17.5*cm, 25.65*cm, id='F1',showBoundary=0)],
                                onPage=partial(header, content=Paragraph('Concept pdf',T2)))  # onPage=partial(...) doesn't need to be changed, uses the header function
        self.addPageTemplates(template)
        
    def afterFlowable(self, flowable):  # needed for the table of contents
        #"Registers TOC entries."
        if flowable.__class__.__name__ == 'Paragraph':
            text = flowable.getPlainText()
            style = flowable.style.name
            if style == 'Heading1':
                key = 'h1-%s' % self.seq.nextf('Heading1')
                self.canv.bookmarkPage(key)
                self.notify('TOCEntry', (0, text, self.page,key))
            if style == 'Heading2':
                key = 'h2-%s' % self.seq.nextf('Heading2')
                self.canv.bookmarkPage(key)
                self.notify('TOCEntry', (1, text, self.page, key))

def header(canvas, doc, content):
    canvas.saveState()
    w, h = content.wrap(doc.width, doc.topMargin)
    canvas.setStrokeColorRGB(0/256, 190/256, 224/256)  # Color for botom line (footer)
    canvas.line(1.78*cm, 1.78*cm,(21 - 1.78)*cm,1.78*cm)
    canvas.setFillColorRGB(0/256, 190/256, 224/256)   # Color for footer content
    canvas.drawString( 1.78*cm, (0.2 * inch)+ 7*mm,"Concept pdf")   # Footer content
    canvas.drawRightString((21 - 1.78)*cm,(0.2 * inch)+ 7*mm,"%d" % doc.page)  # Page numbers (also footer content)
    canvas.setFillColorRGB(0/256, 190/256, 224/256)
    canvas.setStrokeColorRGB(0/256, 155/256, 224/256)  # Color for topline (header)
    canvas.line(1.78*cm, 29.7*cm - 65,(21 - 1.78)*cm,29.7*cm - 65)  # Top line (header)
    canvas.restoreState()
    
    
# TOC, h1, h1_blue, h2, T1, T2 all custom letter styles. Use h1, h1_blue, or h2 to add to the table of contents    
TOC = PS(fontname = 'arial-bold',name = 'TOC',fontSize = 15,leading = 16, alignment = 0,
    leftIndent = 0, spaceAfter = 18,textColor = Color(0/256, 155/256, 224/256))

h1 = PS(fontname = 'arial-bold',name = 'Heading1', fontSize = 11, leading = 14)
h1_blue = PS(fontname = 'arial-bold',name = 'Heading1', fontSize = 15, leading = 14, textColor = Color(0/256, 155/256, 224/256))
h2 = PS(fontname = 'arial',name = 'Heading2', fontSize = 10, leading = 14, leftIndent = 0)

T1 = PS(fontname = 'arial',name = 'text',fontsize = 11, leading = 14, alignment = 0)
T2 = PS(fontname = 'arial',name = 'text',fontsize = 11, leading = 14, alignment = 0,color = Color(0/256, 190/256, 224/256))

gh_blue = PS(fontname = 'arial',name = 'Graph-header', fontSize = 11, leading = 11, textColor = Color(0/256, 155/256, 224/256))

def Write(df, pdf_name, path_plot="", choice=[], top_choice=0, critical=0, option=0, AES=0):
    out = path_plot
    out_raw = outpath_raw

    doc = MyDocTemplate("%s.pdf" % pdf_name)
    width, height = A4

    Story = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Justify", alignment=TA_JUSTIFY))

    toc = TableOfContents()
    toc.levelStyles = [
        PS(fontName='arial', fontSize=11, name='TOCHeading1', leftIndent=20, firstLineIndent=-20, spaceBefore=0,
           leading=16),
        PS(fontSize=10, name='TOCHeading2', leftIndent=40, firstLineIndent=-20, spaceBefore=0, leading=12),
    ]

    Story.append(Paragraph('<b>Table of contents</b>', TOC))
    Story.append(toc)
    Story.append(PageBreak())

    #df_diff, df_inc = First_difference(df)

    # All + Product Dataframe list - Sandeep K Edit
    df_tac_diff, df_tac_inc = First_difference(df_tac)
    df_diff = []
    df_inc = []
    for l in range(len(df)):
        df_temp = df[l].copy()
        df_temp_diff, df_temp_inc = First_difference(df_temp)
        df_diff.append(df_temp_diff)
        df_inc.append(df_temp_inc)
    # End of Edit

    crit_neg = [-0.2, -0.4, -0.6, -0.8, -1]
    crit_pos = [0.2, 0.4, 0.6, 0.8, 1]
    top, cut = 10, -6
    TOP_old, TOP_new, TOP_old_diff, TOP_new_diff = Top(df_tac, cutoff)
    rank, rank_top, rank_rest, rank_left = Rank(df_tac, top, TOP_old, TOP_new)
    crit_over, crit_count = Count(df_tac_inc, crit_neg, crit_pos, cutoff=cut)

    df_cut_old = Cut(df_tac, cut, TOP_old, new=0)
    df_cut_new = Cut(df_tac, cut, TOP_new, new=1)

    dist = []
    dist_round = []
    for l in range(len(df)):
        df_temp = df[l].copy()
        dist_temp, dist_round_temp = Dist(df_temp, rank['new'])
        dist.append(dist_temp)
        dist_round.append(dist_round_temp)
    dist_tac, dist_tac_round = Dist(df_tac, rank['new'])

    name = rank.index
    if len(choice) > 0:
        name = choice
    if top_choice > 0:
        name = rank['new'].index[:top_choice]
    if critical > 0:
        crit_now = crit_over.iloc[-1]
        name = (crit_now[crit_now > critical].index).append(crit_now[crit_now < -critical].index).unique()
    if AES > 0:
        name = rank_same_AES.index

    Story.append(Paragraph('<b>Importance of bricks for Tacrolimus</b>', h1_blue))
    # Story.append(Paragraph(rank_header, h1))
    Story.append(Spacer(1, 10))

    rank_text = ['- Bricks: brick number',
                 '- New Top: rank of the bricks based on the new past 6 months',
                 '- Old Top: rank of the bricks based on the previous past 6 months']
    rank_added = 'The bricks that joined the new top 10 are:'
    rank_subtracted = 'The bricks that left the new top 10 are:'
    ws = '      '  # White space

    for i in range(len(rank_text)):
        Story.append(Paragraph(rank_text[i], T1))
    Story.append(Spacer(1, 2))

    Story.append(Paragraph(rank_added, T1))
    for i in range(len(TOP_new_diff)):
        Story.append(XPreformatted('%s- %s' % (ws, TOP_new_diff[i]), T1))
    Story.append(Paragraph(rank_subtracted, T1))
    for i in range(len(TOP_old_diff)):
        Story.append(XPreformatted('%s- %s' % (ws, TOP_old_diff[i]), T1))
    Story.append(Spacer(1, 2))

    logo_rank = outpath_plot_tables + "Rank AES.png"
    Story.append(Image(logo_rank, 5 * inch, 5 * inch))

    AES_added = rank_AES.index[0]
    for i in range(len(rank_AES.index) - 1):
        AES_added = AES_added + ', ' + rank_AES.index[i + 1]
    AES_added_top = rank_AES_top.index[0]
    for i in range(len(rank_AES_top.index) - 1):
        AES_added_top = AES_added_top + ', ' + rank_AES_top.index[i + 1]
    text_AES_added = 'The following bricks have been added the data as possible top 10 bricks: %s' % AES_added
    text_AES_added_top = 'Out of these bricks, the ones belonging to the top are: %s' % AES_added_top
    Story.append(Paragraph(text_AES_added + '.', T1))
    Story.append(Paragraph(text_AES_added_top + '.', T1))
    Story.append(Spacer(1, 10))

    Story.append(PageBreak())
    tot_inc = int(df_tac_all_diff.iloc[-1][0])
    if np.abs(df_tac_all_inc.iloc[-1][0]) > 0.2:
        tot_crit = '(%s,Critical!)' % ('{percent:.2%}'.format(percent=df_tac_all_inc.iloc[-1][0]))
    else:
        tot_crit = '(%s)' % ('{percent:.2%}'.format(percent=df_tac_all_inc.iloc[-1][0]))
    tot_avg_inc = int(df_tac_all_mean_inc)
    tot_avg_inc_perc = '{percent:.2%}'.format(percent=df_tac_all_mean_inc_perc[0])
    text = 'The total counting units increased by %s %s units. Its mean increased by %s (%s) units.' % (tot_inc,
                                                                                                        tot_crit,
                                                                                                        tot_avg_inc,
                                                                                                        tot_avg_inc_perc)

    if os.path.exists(outpath_raw_national + 'Nederland.png') == False:
        logo_all_raw = outpath_raw_national + 'Nederland .png'
    else:
        logo_all_raw = outpath_raw_national + 'Nederland.png'
    Story.append(Paragraph('<b>National level volume development for tacrolimus</b>', h1_blue))
    Story.append(Spacer(1, 10))

    CI_exceeded = ['Counting units confidence interval exceeded!',
                   'Product shares moving quarters confidence interval exceeded!']

    if df_tac_all_over_bound.iloc[0][0] == 1:
        Story.append(Paragraph(CI_exceeded[0], T1))
    #Uncomment below after definition - Sandeep K
    #if df_tac_all_rolling_shares_over_bound.iloc[0][0] == 1:
    #    Story.append(Paragraph(CI_exceeded[1], T1))

    Story.append(Spacer(1, 10))
    # Story.append(Image(logo_all_raw,7.5*inch,3.5*inch,hAlign = 'LEFT'))
    Story.append(Image(logo_all_raw, 7 * inch, 3.25 * inch, hAlign='LEFT'))
    Story.append(Spacer(1, 10))

    # Uncomment after definiton of 'out_comb_national' - Sandeep K
    #outpath_edge_nederland = out
    #if os.path.exists(outpath_edge_nederland + 'Nederland.png') == False:
    #    logo_all_edge = outpath_edge_nederland + 'Nederland .png'
    #else:
    #    logo_all_edge = outpath_edge_nederland + 'Nederland.png'
    #Story.append(Spacer(1, 2))
    ## Story.append(Image(logo_all_edge,7.5*inch,2.25*inch,hAlign='LEFT'))
    #Story.append(Image(logo_all_edge, 7.2 * inch, 2.25 * inch, hAlign='LEFT'))
    #Story.append(Spacer(1, 20))
    #Story.append(Paragraph(text, T1))
    #Story.append(Spacer(1, 10))

    text_CI = 'CI (Confidence interval): If the value of the data exceeds the confidence interval at any point in time it means that that value is not as expected. This value can then be seen as an "extreme" observation and it needs to be looked at.'
    Story.append(Paragraph(text_CI, T1))
    Story.append(Spacer(1, 5))

    if len(df_tac_over_bound_brick) > 0:
        over = df_tac_over_bound_brick[0]
        if len(df_tac_over_bound_brick) > 1:
            for i in range(len(df_tac_over_bound_brick) - 1):
                over = over + ', ' + df_tac_over_bound_brick[i + 1]
        over_CI_text1 = '<b>Note!</b>: The following bricks have exceeded their confidence interval: %s' % over
        Story.append(Paragraph(over_CI_text1, T1))

    Story.append(PageBreak())

    Story.append(Paragraph('<b>ALL ALERTS</b>', h2))

    Story.append(PageBreak())

    print(name)    #QC Check - Sandeep K
    num_of_top_bricks = len(name)
    for abnormal_brickname in df_tac_over_bound_brick:
        name = name.insert(len(name), abnormal_brickname)

    for i in range(len(name)):

        for k in range(len(df)):    # Added iterator for product level data - Edit by Sandeep K

            if i == 0 and k == 0:
                Story.append(Paragraph('<b> TOP COMMON BRICKS BY SALES </b>', h1_blue))
                Story.append(PageBreak())
                styles = getSampleStyleSheet()

            if i == num_of_top_bricks and k == 0:
                Story.append(Paragraph('<b> BRICKS EXCEEDING LIMITS </b>', h1_blue))
                Story.append(PageBreak())
                styles = getSampleStyleSheet()

            # Update similarly for Product names (against brick), Images, Rolling qtr shares and table inputs.
            #if i == 0:
            Story.append(Paragraph('<b>Brick (' + name[i][:2] + ') level volume development for ' + PRD[k] + '</b>', h1_blue))
            Story.append(Spacer(1, 10))

            AES_name = ''
            for j in range(len(brick_AES)):
                if brick_AES[j] == name[i].split(" ")[0]:
                    AES_name = ' (AES) '

            header = 'Brick: %s%s' % (name[i], AES_name)

            Story.append(Paragraph(header, h2))
            Story.append(Spacer(1, 5))

            ranking = 'Rank: %s (Tacrolimus mol-level)' % dist_tac['Rank'].loc[name[i]]

            Story.append(Paragraph(ranking, T1))
            Story.append(Spacer(1, 5))

            tabs_CI_abs, tabs_CI_shares = 'no', 'no'

            if df_tac_over_bound[name[i]][0] == 1:
                Story.append(Paragraph('%s' % CI_exceeded[0], T1))
                tab_CI_abs = 'yes'
            # Uncomment below after definition - Sandeep K
            #if df_tac_rolling_shares_over_bound[name[i]][0] == 1:
            #    Story.append(Paragraph('%s' % CI_exceeded[1], T1))
            #    tabs_CI_shares = 'yes'
            #    Story.append(Spacer(1, 10))

            # Graph Titles added - Sandeep K Edit
            #Story.append(Spacer(1, 4))
            #graph_header_1 = 'Market graph (Gram vs. Time):'
            #Story.append(Paragraph(graph_header_1, gh_blue))
            #Story.append(Spacer(1, 4))

            if os.path.exists(out_raw[k] + name[i] + ".png") == False:
                logo1 = r"%s%s .png" % (out_raw[k], name[i])
            else:
                logo1 = r"%s%s.png" % (out_raw[k], name[i])

            width, height = A4
            # im1 = Image(logo1,7.5*inch,3.5*inch)
            im1 = Image(logo1, 7 * inch, 3.25 * inch)
            im1.hAlign = 'LEFT'

            Story.append(im1)

            # Graph Titles added - Sandeep K Edit and Removed
            #graph_header_2_1 = 'Absolute market shares graph (Gram vs. Time):'
            #graph_header_2_2 = 'Rolling quarters shares graph (Market shares vs. Time):'
            #Story.append(Paragraph(graph_header_2_1 + ' ' + graph_header_2_2, gh_blue))
            #Story.append(Paragraph(graph_header_2_2, gh_blue))
            #Story.append(Spacer(1, 4))

            if os.path.exists(out[k] + name[i] + ".png") == False:
                logo2 = r"%s%s .png" % (out[k], name[i])
            else:
                logo2 = r"%s%s.png" % (out[k], name[i])

            # im2 = Image(logo2,7.5*inch,2.25*inch,hAlign='LEFT')
            im2 = Image(logo2, 7.2 * inch, 2.25 * inch, hAlign='LEFT')

            Story.append(Spacer(1, 8))

            Story.append(im2)
            Story.append(Spacer(1, 20))

            text = 'The graph on the left shows the last 12 months of the Counting Units within the brick. The graph on the right shows the the product shares moving quarters. Both graphs include a forecast (hard green line) and its confidence interval (gray vertical line). The confidence intervals are given by the blue dotted lines. The red dotted line is the mean calculated last month, the green dotted line is the mean calculated this month.'
            t_inc = ['increased', 'decreased']

            s_inc, m_inc = 0, 0
            #print('i='+str(i)+', k='+str(k))    #QC Check
            #print(df_inc[k])    #QC Check
            #print(dist[k])    #QC Check
            if df_inc[k][name[i]][-1] < 0:
                s_inc = 1
            if dist[k]['% increase'][name[i]] < 0:
                m_inc = 1

            Story.append(Paragraph(text, T1))
            Story.append(Spacer(1, 10))

            cu_delta = df_diff[k][name[i]][-1]
            if numpy.isnan(cu_delta):
                cu_delta = 'NA'
            else:
                cu_delta = int(cu_delta)
            text = 'The counting units %s by %s units (%s).' % (
            t_inc[s_inc], cu_delta, perc(df_inc[k][name[i]][-1]))
            # Removed int() for 2nd argument to support NaNs (for now)
            cu_mean = dist[k]['Mean diff'][name[i]]
            if numpy.isnan(cu_mean):
                cu_mean = 'NA'
            else:
                cu_mean = int(cu_mean)
            text = text + ' Its mean %s by %s units (%s)' % (
            t_inc[m_inc], cu_mean, perc(dist[k]['% increase'][name[i]]))
            # Removed int() for 2nd argument to support NaNs (for now)

            Story.append(Paragraph(text, T1))

            s_inc, m_inc = 0, 0
            if df_rolling_shares_inc[k][name[i]][-1] < 0:
                s_inc = 1
            if dist_rolling_shares[k]['% increase'][name[i]] < 0:
                m_inc = 1

            shr_delta = round(df_rolling_shares_diff[k][name[i]][-1], 5)
            if numpy.isnan(shr_delta):
                shr_delta = 'NA'
            text = 'The product shares of the moving quarters counting units %s by %s (%s).' % (
            t_inc[s_inc], shr_delta, perc(df_rolling_shares_inc[k][name[i]][-1]))
            shr_mean = round(dist_rolling_shares[k]['Mean diff'][name[i]], 5)
            if numpy.isnan(shr_mean):
                shr_mean = 'NA'
            text = text + ' Its mean %s by %s (%s)' % (t_inc[m_inc], shr_mean,
                                                       perc(dist_rolling_shares[k]['% increase'][name[i]]))

            Story.append(Paragraph(text, T1))

            text = ''

            Story.append(Spacer(1, 10))
            T3 = PS(fontname='arial', name='text', fontsize=11, leading=14, alignment=0, color=Color(1, 1, 1))

            # Create a table at the bottom of each BRICK page
            h = [Paragraph('<font color=white><b>Old \nmean</b></font>', T3),
                 Paragraph('<font color=white><b>New \nmean</b></font>', T3),
                 Paragraph('<font color=white><b>CI\nexceeded</b></font>', T3)]
            table_data = [['', h[0], h[1], h[2]],
                          ['Absolute', int(dist[k]['Mean old top'][name[i]]), int(dist[k]['Mean new top'][name[i]]), tabs_CI_abs],
                          ['Shares', round(dist_rolling_shares[k]['Mean old top'][name[i]], 4),
                           round(dist_rolling_shares[k]['Mean new top'][name[i]], 4), tabs_CI_shares]]

            table = Table(table_data, (2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm), hAlign='CENTER',
                          style=[('BOX', (0, 0), (-1, -1), 0.5, colors.black),
                                 ('GRID', (0, 0), (-1, -1), 0.5, colors.black),  # Style the table
                                 ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0 / 256, 163 / 256, 224 / 256)),
                                 ('TEXTCOLOR', (0, 0), (1, 0), colors.white), ('TEXTCOLOR', (2, 0), (3, 0), colors.white),
                                 ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                                 ('FONTNAME', (0, 0), (-1, -1), 'arial'), ('FONTSIZE', (0, 0), (-1, -1), 11)])
            Story.append(table)
            if ((i + 1) % 1) == 0:
                Story.append(PageBreak())

            styles = getSampleStyleSheet()

    doc.multiBuild(Story)
    return print('done')

Write(df, pdf_name="Output_results", path_plot=outpath_comb, top_choice=2, option=1, AES=1)

#---------------------------------------------------------------------------
# Create the cover page
background = cwd + "PDF cover\\IQVIA_cover.png"
pdf=canvas.Canvas("Front_cover.pdf",pagesize=A4)
pdf.drawImage(background, 0, 0, 8.27*inch, 11.69*inch)
pdf.translate(cm, cm)
pdf.setFontSize(26)
pdf.setFont('arial-bold',26)
x_title, y_title = 490,685
pdf.drawRightString(x_title,y_title,"Early Warning report")  #title of the cover page

pdf.setFont('arial-italic',26)
x_subtitle, y_subtitle = x_title, y_title - 35

pdf.setFillColorRGB(0/256, 155/256, 224/256)
subtitle = ''    # Subtitle of the cover page
if len(subtitle) == 0:
    x_date, y_date = x_subtitle, y_subtitle
else:
    x_date,y_date = x_subtitle, y_subtitle-25

pdf.drawRightString(x_subtitle,y_subtitle,"")
pdf.setFont('arial',12)
pdf.setFillColorRGB(0,0,0)
pdf.drawRightString(x_date,y_date,"Month %s, %s"%(end_date[-2:],end_date[-6:-2]))   # Date of the cover page
pdf.rotate(20)
pdf.save()

# Merge cover page and results
pdfs = ['Front_cover.pdf', 'Output_results.pdf']
merger = PdfFileMerger()
for pdf in pdfs:
    merger.append(pdf)

merger.write("result.pdf")
merger.close()



print('PDF finished')
