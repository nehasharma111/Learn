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
