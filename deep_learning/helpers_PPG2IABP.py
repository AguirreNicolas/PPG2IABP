import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
import pandas as pd
import copy


#Filters
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy import signal
from scipy.fft import fft
from scipy.signal import savgol_filter, find_peaks


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import norm, pearsonr
from sklearn import linear_model

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, sio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)





def func_x_y_sets(df_o,df_file_name,q_files,len_signal,channels_selected,
                  fs,window_training,max_n_records,
                  q_signals_file,t_size,r_seed,
                  sbp_threshold,duration_threshold,skew_threshold,
                  post_beat_max,penalize_max_t,
                  lowcut,highcut,order,
                  window,ploy_grade,s_mode,dx_n,
                  mixed,
                  signals_path,all_plots=False):
  
  # FROM VISUAL CONTROL WAS DETERMINED THAT THE MARKER TREND TO DELAY THE
  # DICROTIC NOTCH POIT.
  shift_dn = -2

  df_f = df_o.copy()
  df_f = df_f.reset_index(drop=True)
  if q_signals_file > 1:
    df_f = pd.concat([df_f] * q_signals_file, ignore_index=False)
  df_f = df_f.sort_index()

  input_channels = ['PLETH']
  input_channels_dict = {k: v for v, k in enumerate(input_channels)}
  input_channels = [channels_selected.get(key) for key in input_channels]
  # X
  x = np.zeros((q_files*q_signals_file,len(input_channels),len_signal),dtype=float)
  # Y
  dim_y = 2*fs+1 #lenght of mean pulse (FROM MATLAB STRUCT)
  y = np.zeros((q_files*q_signals_file,dim_y),dtype=float)
  sys_idx = np.zeros((q_files*q_signals_file,1),dtype=int)
  dn_idx = np.zeros((q_files*q_signals_file,1), dtype=int)
  end_idx = np.zeros((q_files*q_signals_file,1), dtype=int)
  sys_val = np.zeros((q_files*q_signals_file,1),dtype=float)
  dn_val = np.zeros((q_files*q_signals_file,1), dtype=float)
  end_val = np.zeros((q_files*q_signals_file,1), dtype=float)
  durations = np.zeros((q_files*q_signals_file,1), dtype=int)
  skew = np.zeros((q_files*q_signals_file,1), dtype=float)
  y_sys = np.zeros((q_files*q_signals_file,1), dtype=float)
  y_dbp = np.zeros((q_files*q_signals_file,1), dtype=float)
  #window for mark_corrections
  w_adjust = 5

  # Creation of X and Y set´s
  for i_file in np.arange(q_files):
    #Load all the file
    if q_signals_file > 1:
      file = loadmat(signals_path+df_f.loc[i_file,'subrecord'].array[0])
    else:
      file = loadmat(signals_path+df_f.loc[i_file,'subrecord'])
    #if not (i_file % 1000):
    #  print(i_file)    
    file = file['signal_processing']
    # Index1
    idx_file = i_file*q_signals_file
    for i_signal in np.arange(0,q_signals_file):
      # Index for X set
      idx_signal = idx_file + i_signal
      # X
      x[idx_signal,:,:] = file[i_signal].signal[input_channels,:] #signal[[channels of interes],:]
      ## Mean ABP pulse
      #Signals
      y[idx_signal,:] = file[i_signal].struct_features.mean_pulse_abp
      y_sys[idx_signal] = y[idx_signal,:].max() 
      y_dbp[idx_signal] = y[idx_signal,:].min()
      #y_dn[idx_signal] = y[idx_signal,:].max()
      skew[idx_signal] = file[i_signal].struct_stats_features.abp.mean.skew
      #Marks
      sys_idx[idx_signal] = int(file[i_signal].struct_stats_features.abp.mean.iTTSP)
      values_syst = y[idx_signal,sys_idx[idx_signal,0]-w_adjust:sys_idx[idx_signal,0]+w_adjust]
      corrected_syst = np.argmax(values_syst)
      sys_idx[idx_signal] = sys_idx[idx_signal,0] + corrected_syst - w_adjust
      dn_idx[idx_signal] = int(file[i_signal].struct_stats_features.abp.mean.iLVET)
      end_idx[idx_signal] = int(file[i_signal].struct_stats_features.abp.mean.iHP)
      sys_val[idx_signal] = y[idx_signal,sys_idx[idx_signal]]
      dn_val[idx_signal] = y[idx_signal,dn_idx[idx_signal]+shift_dn]
      end_val[idx_signal] = (y[idx_signal,end_idx[idx_signal]] + y[idx_signal,0])/2
      durations[idx_signal] = np.where(y[idx_signal,:] <= 40)[0][0]-1


  # Check Skew
  check_skew = np.where(skew > skew_threshold )[0]
  if all_plots:
    plt.hist(skew,bins=20)
    plt.show()
  #print(f'Ok Skew: {np.shape(check_skew)[0]}')
  x = x[check_skew,:,:]
  y = y[check_skew,:]
  skew = skew[check_skew]
  sys_idx = sys_idx[check_skew]
  dn_idx = dn_idx[check_skew]
  end_idx = end_idx[check_skew]
  sys_val = sys_val[check_skew]
  dn_val = dn_val[check_skew]
  end_val = end_val[check_skew]
  durations = durations[check_skew]
  df_f = df_f.iloc[check_skew,:]
  df_f.reset_index(drop=True,inplace=True)
  y_dbp = y_dbp[check_skew]
  y_sys = y_sys[check_skew]

  # Check Durations
  check_end_idx = np.where(end_idx < duration_threshold )[0]
  if all_plots:
    plt.hist(end_idx,bins=20)
    plt.show()
  #print(f'Ok Durations: {np.shape(check_end_idx)[0]}')

  x = x[check_end_idx,:,:]
  y = y[check_end_idx,:]
  skew = skew[check_end_idx]
  sys_idx = sys_idx[check_end_idx]
  dn_idx = dn_idx[check_end_idx]
  end_idx = end_idx[check_end_idx]
  sys_val = sys_val[check_end_idx]
  dn_val = dn_val[check_end_idx]
  end_val = end_val[check_end_idx]
  durations = durations[check_end_idx]
  df_f = df_f.iloc[check_end_idx,:]
  df_f.reset_index(drop=True,inplace=True)
  y_dbp = y_dbp[check_end_idx]
  y_sys = y_sys[check_end_idx]

  # Check SBP
  if all_plots:
    n_bins=50
    alpha_v=0.5
    plt.hist(sys_val, n_bins, facecolor='r', alpha=alpha_v,label='SBP')
    plt.hist(end_val, n_bins, facecolor='b', alpha=alpha_v,label='DBP')
    plt.xlabel('Pressure[mmHg]',fontsize=18)
    plt.ylabel('Counts',fontsize=18)
    plt.legend()
    plt.title('Histogram of Systolic & Diastolic Pressure',fontsize=18)
    plt.show()

  check_sys_val = np.where(sys_val < sbp_threshold )[0]
  #print(f'Ok Syst. Values: {np.shape(check_sys_val)[0]}')

  x = x[check_sys_val,:,:]
  y = y[check_sys_val,:]
  skew = skew[check_sys_val]
  sys_idx = sys_idx[check_sys_val]
  dn_idx = dn_idx[check_sys_val]
  end_idx = end_idx[check_sys_val]
  sys_val = sys_val[check_sys_val]
  dn_val = dn_val[check_sys_val]
  end_val = end_val[check_sys_val]
  durations = durations[check_sys_val]
  df_f = df_f.iloc[check_sys_val,:]
  df_f.reset_index(drop=True,inplace=True)
  y_dbp = y_dbp[check_sys_val]
  y_sys = y_sys[check_sys_val]

  x_shape = np.shape(x)
  y_shape = np.shape(y)
  q_signals = y_shape[0]

  #print(f'Shape X: {x_shape}  Y: {y_shape}')  
  #############################
  # Y adjusted + Mask
  #############################
  max_dur = end_idx.max()
  min_dur = end_idx.min()  
  y_adjusted = np.zeros((q_signals,1,max_dur+post_beat_max))
  y_mask = np.ones((q_signals,1,max_dur+post_beat_max))

  foot_roll = 0 #If you what to roll the singnal change it.
  sys_idx = sys_idx - foot_roll
  dn_idx = dn_idx - foot_roll
  end_idx = end_idx - foot_roll

  for i_s in np.arange(0,q_signals):
    end = end_idx[i_s,0]
    signal_0 = y[i_s,:end]
    repeatetitions = np.tile(signal_0,3*int(np.ceil(max_dur/end)))
    signal_cat = np.concatenate((signal_0,repeatetitions))
    signal_cat = np.roll(signal_cat,-foot_roll)
    y_adjusted[i_s,0,:] = signal_cat[:max_dur+post_beat_max]
    y_mask[i_s,0,end+penalize_max_t:] = 0  

  #############################
  # Filter & Scaling
  #############################
  ######### PPG ###############
  scaler = MinMaxScaler()
  x_filt = copy.deepcopy(x)
  x_filt_norm = copy.deepcopy(x)
  x_d1_filt_norm = np.zeros_like(x_filt_norm)

  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  b, a = signal.butter(order, [low, high], btype='band')
  
  norm_channels = ['PLETH']
  norm_channels = [input_channels_dict.get(key) for key in norm_channels]
  for i_x , i_c in enumerate(norm_channels):
      for i_s in np.arange(0,x_shape[0]):
          x_filt[i_s,0,:] = signal.filtfilt(b,a,x[i_s,0,:])       
          x_filt_norm[i_s,0,:] = scaler.fit_transform(x_filt[i_s,0,:].reshape(-1,1)).reshape(-1,)
          x_d1_filt_norm[i_s,0,:]  = savgol_filter(x_filt_norm[i_s,0,:],window,ploy_grade,mode=s_mode,deriv=dx_n)
          x_d1_filt_norm[i_s,0,:]  = scaler.fit_transform(x_d1_filt_norm[i_s,0,:].reshape(-1,1)).reshape(-1,)
  ######### ABP ###############
  y_norm_adjusted = copy.deepcopy(y_adjusted)
  #Limits
  max_abp = 100
  min_abp = 100
  for i_s in np.arange(0,q_signals):
    max_i = np.max(y_norm_adjusted[i_s,0,:])
    min_i = np.min(y_norm_adjusted[i_s,0,:])
    if min_i < 1:
      print(i_s)
    if max_i > max_abp:
      max_abp = max_i
    if min_i < min_abp:
      min_abp = min_i
  lim_P = [min_abp,max_abp]
  #print(f'Min & Max ABP: {lim_P}')

  #Dictionary with lim
  norm_dict = {'P':[lim_P[0],lim_P[1]]}
  #Scalers Obj.
  scaler_P = MinMaxScaler()
  scaler_P.fit(np.array(norm_dict['P'])[:, np.newaxis])
  for i_s in np.arange(0,q_signals):
    y_norm_adjusted[i_s,0,:] = scaler_P.transform(y_norm_adjusted[i_s,0,:].reshape(-1,1)).reshape(-1,)
  
  #Delet 1 seg at the begin and at the end
  borders = fs
  x_filt_norm = x_filt_norm [:,:,borders:-borders]
  x_d1_filt_norm = x_d1_filt_norm [:,:,borders:-borders]

  if all_plots:
    sgn=np.random.randint(0,high=x_shape[0])
    plt.plot(x_filt_norm[sgn,0,:])
    plt.plot(x_d1_filt_norm[sgn,0,:])
    plt.show()

  #############################
  # CONCATENATING PPG & PPG'
  #############################
  x_signal_dx = np.concatenate((x_filt_norm,x_d1_filt_norm),axis=1)
  # Signal duration
  len_signal_ppg = np.shape(x_signal_dx)[2]
  #Max beat duration
  final_len_y = max_dur + post_beat_max
  final_len_x = fs*window_training
  diff_window_ppg = len_signal_ppg - final_len_x
  #print(f'Final len X: {final_len_x} | Final len Y: {final_len_y}')
  #############################
  # SEGMENTATION
  #############################
  segment_matrix =np.zeros((np.shape(y_norm_adjusted)[0],1,final_len_y))
  if all_plots:
    sgn = np.random.randint(0,high=q_signals)
    plt.plot(np.arange(0,sys_idx[sgn,0]+1)/125,y_norm_adjusted[sgn,0,:sys_idx[sgn,0]+1],'r',label='Onset - systolic peak')
    plt.plot(np.arange(sys_idx[sgn,0],dn_idx[sgn,0]+shift_dn+1)/125,y_norm_adjusted[sgn,0,sys_idx[sgn,0]:dn_idx[sgn,0]+shift_dn+1],'g',label='Systolic peak - dicrotic notch')
    plt.plot(np.arange(dn_idx[sgn,0]+shift_dn,end_idx[sgn,0]+1)/125,y_norm_adjusted[sgn,0,dn_idx[sgn,0]+shift_dn:end_idx[sgn,0]+1],'b',label='Dicrotic notch - end')
    plt.plot(np.arange(end_idx[sgn,0],final_len_y)/125,y_norm_adjusted[sgn,0,end_idx[sgn,0]:final_len_y],'k',label='Ended')
    plt.vlines((end_idx[sgn,0]+penalize_max_t)/125,1, 0, linestyles ="dashed", colors ="m",label='Mask limit')
    plt.legend(loc='upper right',fontsize=8)
    plt.xlabel('Time [s]',fontsize=18)
    plt.ylabel('Scaled Blood Pressure',fontsize=18)
    plt.ylim((0,1))
    plt.title('$\overline{ABPM}$ for training',fontsize=20)
    #plot_name = str(plot_path+'output_model'+".eps")
    #plt.savefig(plot_name, dpi=300,bbox_inches = "tight")
    plt.show()
  segment_matrix =np.zeros((np.shape(y_norm_adjusted)[0],4,final_len_y))
  for i in np.arange(0,q_signals):
    segment_matrix[i,1,:sys_idx[i,0]] = 1
    segment_matrix[i,2,sys_idx[i,0]:dn_idx[i,0]+shift_dn] = 1
    segment_matrix[i,3, dn_idx[i,0]+shift_dn:end_idx[i,0]] = 1
    segment_matrix[i,0, end_idx[i,0]:] = 1
  y_norm_adjusted_segments = np.concatenate((y_norm_adjusted[:,:,:final_len_y],segment_matrix),axis=1)

  #############################
  # JOIN SBP/DBP/DNBP VALUES & INDEX
  ############################# 
  y_values_bp = np.concatenate((sys_val,dn_val,end_val),axis=1)
  y_idx_bp = np.concatenate((sys_idx,dn_idx+shift_dn,end_idx),axis=1)

  #############################
  # LIMIT N° OF SIGNALS
  #############################
  max_repetitions = max_n_records * q_signals_file
  repetitions_df_f = df_f.groupby('subject_id').cumcount().reset_index()
  repetitions_df_f['subject_id'] = df_f['subject_id']
  index_rep_df_f = repetitions_df_f[repetitions_df_f[0]<max_repetitions].index.values

  x_signal_dx = x_signal_dx[index_rep_df_f]
  y_norm_adjusted_segments = y_norm_adjusted_segments[index_rep_df_f]
  y_mask = y_mask[index_rep_df_f]
  y_values_bp = y_values_bp[index_rep_df_f]
  y_idx_bp = y_idx_bp[index_rep_df_f]

  df_f = df_f.iloc[index_rep_df_f,:].reset_index(drop=True)

  q_files = df_f.shape[0]
  #print(f'N° of signals after limit n_records: {q_files}')
  if all_plots:
    repetitions_array = df_f['subject_id'].value_counts(ascending=False)
    repetitions_array = repetitions_array.value_counts(ascending=False).reset_index().sort_values(by=['index']).values
    sum_signals = repetitions_array[:,0] * repetitions_array[:,1]
    cum_sum_signals = np.cumsum(sum_signals)/q_files

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Number of segments',fontsize=18)
    ax1.set_ylabel('Number of subjects',fontsize=18)
    ax1.bar(repetitions_array[:,0],repetitions_array[:,1],color='grey',label = 'Number of subjects' )
    ax1.tick_params(axis='y',)
    plt.legend(loc="upper left",fontsize=12,bbox_to_anchor=(0.25,1))
    #plt.title("Count of subjects per segments quantities and \n segments accumulative percentage",fontsize=14)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('%', color='k')  # we already handled the x-label with ax1
    ax2.plot(repetitions_array[:,0],cum_sum_signals, color='r',label='Accum. of segments (%)')
    ax2.hlines(y=0.5,xmin=0,xmax=11,linestyles='dashed')
    ax2.set_xlim((0.5,max_repetitions+0.5))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend(loc="upper left",fontsize=12,bbox_to_anchor=(0.25,0.85))
    #plot_name = str(plot_path+'Accumulative_no_bias'+".eps")
    #plt.savefig(plot_name, dpi=300,bbox_inches = "tight")
    plt.show()
    #print(f'Qty subject > 50%:\r\n{repetitions_array[cum_sum_signals>0.5][:,1].sum()} subjects with {sum_signals[cum_sum_signals>0.5].sum()} signals')

  #############################
  #SPLIT
  #############################
  if mixed:
    encoder_ci = False
    X_ci_train_scaled,X_ci_test_scaled, df_train, df_test = split_scale_df(df_f,df_file_name,final_len_x,t_size,r_seed,encoder=encoder_ci)
    x_train, x_test, y_train, y_test,y_mask_train,y_mask_test  = train_test_split(x_signal_dx, y_norm_adjusted_segments,y_mask, test_size=t_size, random_state=r_seed)
    # BP Values
    y_values_bp_train, y_values_bp_test = train_test_split(y_values_bp, test_size=t_size, random_state=r_seed)
    # Index
    y_idx_bp_train, y_idx_bp_test = train_test_split(y_idx_bp, test_size=t_size, random_state=r_seed)

    q_train = np.shape(x_train)[0]
    q_test = np.shape(x_test)[0]
    #print(f'Train: {q_train} | Test: {q_test}')
  else:
    ids_uniques = df_f.subject_id.unique()
    ids_unique_train, ids_uniques_test  = train_test_split(ids_uniques, test_size=t_size, random_state=r_seed)
    #print(f'subjects_train: {ids_unique_train.shape} | subjects_test: {ids_uniques_test.shape}')

    df_train = df_f.loc[df_f['subject_id'].isin(ids_unique_train),:]
    df_test = df_f.loc[df_f['subject_id'].isin(ids_uniques_test),:]
    idx_train = df_train.index
    idx_test = df_test.index

    #Standarize Clinical Info
    if df_file_name == 'sex_age':
      numerical_col = ['age']
    if df_file_name == 'weight_height':
      numerical_col = ['age','height','weight']

    n_numerical_col = len(numerical_col)
    categorical_col = ['gender']
    n_categorical_col = len(categorical_col)

    clinic_info = numerical_col+categorical_col
    n_clinic_info = len(clinic_info)
    # Standarize
    scaler = StandardScaler()
    scaler.fit(df_train[numerical_col])
    X_scaled = scaler.transform(df_train[numerical_col])
    X_categorical = pd.get_dummies(df_train,columns=categorical_col).iloc[:,-2:].values #-2 po H =[0,1], M=[1,0]
    X_ci_train_scaled = np.concatenate((X_scaled,X_categorical),axis=1)

    X_scaled = scaler.transform(df_test[numerical_col])
    X_categorical = pd.get_dummies(df_test,columns=categorical_col).iloc[:,-2:].values
    X_ci_test_scaled = np.concatenate((X_scaled,X_categorical),axis=1)

    x_train = x_signal_dx[idx_train]
    x_test = x_signal_dx[idx_test]
    y_train = y_norm_adjusted_segments[idx_train]
    y_test = y_norm_adjusted_segments[idx_test]
    y_mask_train = y_mask[idx_train]
    y_mask_test  = y_mask[idx_test]

    y_values_bp_train = y_values_bp[idx_train]
    y_values_bp_test = y_values_bp[idx_test]
    y_idx_bp_train = y_idx_bp[idx_train]
    y_idx_bp_test = y_idx_bp[idx_test]

    q_train = np.shape(x_train)[0]
    q_test = np.shape(x_test)[0]

  return(x_train,x_test,y_train,y_test,y_mask_train,y_mask_test,
        y_values_bp_train,y_values_bp_test,y_idx_bp_train,y_idx_bp_test,
        X_ci_train_scaled,X_ci_test_scaled, df_train, df_test,
        q_train,q_test,final_len_x,final_len_y,diff_window_ppg,scaler_P)


def split_scale_df(df,df_file_name,final_len_x,t_size,r_seed,encoder=True):
  ### Clinical Info
  X_ci_train, X_ci_test = train_test_split(df.copy(), test_size=t_size, random_state=r_seed)
  df_train = X_ci_train.reset_index(drop=True)
  df_test = X_ci_test.reset_index(drop=True)

  #Standarize Clinical Info
  if df_file_name == 'sex_age':
    numerical_col = ['age']
  if df_file_name == 'weight_height':
    numerical_col = ['age','height','weight']

  n_numerical_col = len(numerical_col)
  categorical_col = ['gender']
  n_categorical_col = len(categorical_col)

  clinic_info = numerical_col+categorical_col
  n_clinic_info = len(clinic_info)
  #print(f'NÂ° Clinic Info:{n_clinic_info} \r\nColumns: {clinic_info}')
  # Standarize
  scaler = StandardScaler()
  scaler.fit(X_ci_train[numerical_col])
  X_scaled = scaler.transform(X_ci_train[numerical_col])
  X_categorical = pd.get_dummies(X_ci_train,columns=categorical_col).iloc[:,-2:].values #-2 po H =[0,1], M=[1,0]
  X_ci_train_scaled = np.concatenate((X_scaled,X_categorical),axis=1)

  X_scaled = scaler.transform(X_ci_test[numerical_col])
  X_categorical = pd.get_dummies(X_ci_test,columns=categorical_col).iloc[:,-2:].values
  X_ci_test_scaled = np.concatenate((X_scaled,X_categorical),axis=1)
  if encoder:
    X_ci_train_scaled = np.expand_dims(X_ci_train_scaled,axis=2)
    X_ci_train_scaled = np.tile(X_ci_train_scaled,(1,1,final_len_x))
    X_ci_test_scaled = np.expand_dims(X_ci_test_scaled,axis=2)
    X_ci_test_scaled = np.tile(X_ci_test_scaled,(1,1,final_len_x))

  return X_ci_train_scaled,X_ci_test_scaled, df_train, df_test

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_attention(x,y_pred,attention_weights,y_true= [], y_labels = [],clrmap = 'jet' ):
  len_x = np.shape(x)[1]
  len_y = len(y_pred)

  fig = plt.figure(figsize=(10,10))
  fig.suptitle("Input, $\widehat{ABPM}$ and  $\overline{ABPM_v}$ with Attention",y=0.95,fontsize=20)
  gs = GridSpec(2, 3, width_ratios=[3, 1,1], height_ratios=[1,3],wspace=0.05,hspace=0.05)
  # y_pred vs y_true
  ax1 = fig.add_subplot(gs[0])
  #ax1.set_title('$\widehat{ABPM}$ and $\overline{ABPM_v}$',fontsize=18)
  if np.array(y_labels).size != 0:
    sys = np.where(y_labels==1)[0][-1]+1
    dn_indexs = np.where(y_labels==2)[0]
    dn_indexs = dn_indexs[dn_indexs > sys][-1]
    ax1.plot(np.arange(0,sys),y_pred[0:sys],'r',label='$C_{[O,SP]}$')
    ax1.plot(np.arange(sys-1,dn_indexs),y_pred[sys-1:dn_indexs],'g',label='$C_{[SP, DN]}$')
    ax1.plot(np.arange(dn_indexs-1,len_y),y_pred[dn_indexs-1:],'b',label='$C_{[DN, E]}$')
    #ax1.legend(ncol=2)
    #ax1 = plot_segments_att(y_pred,y_labels,ax1,colorbar=False)
  else:
    ax1.plot(y_pred,label='y_predicted',color='b',)

  if np.array(y_true).size != 0:
    ax1.plot(y_true,label = '$\overline{ABPM_v}$',color = 'k')
  ax1.legend(loc='upper right',fontsize=14,ncol=2)
  ax1.set_ylim((0,1))
  ax1.set_xlim(0,len(y_pred))
  ax1.set_xticks([])
  ax1.set_ylabel('Scaled Blood Pressure',fontsize=14)

  # Attention
  ax2 = fig.add_subplot(gs[3])
  np.shape(x)
  ax2.matshow(attention_weights,aspect='auto',
    origin = 'upper',cmap=clrmap,interpolation='bilinear',extent = [0. ,len_y/125,len_x/125,0.])
  ax2.xaxis.set_ticks_position('bottom')
  ax2.set_xlabel('Prediction Time [s]',fontsize=18)
  ax2.set_ylabel('Input Time [s]',fontsize=18)
  # Input
  ax3 = fig.add_subplot(gs[4])
  ax3.set_title('Input',x=1,fontsize=18)

  ax3.plot(x[0,:],np.arange(0,len_x),color = 'k',label='PPG')
  ax3.set_ylim(0,len_x)
  ax3.legend(loc='upper right',fontsize=14)
  ax3.invert_yaxis()
  ax3.set_yticks([])
  ax3.set_xlabel('Scaled PPG and PPG',x=1,fontsize=18)
 

  ax3 = fig.add_subplot(gs[5])
  ax3.plot(x[1,:],np.arange(0,len_x),color = 'k',label="PPG'")
  ax3.set_ylim(0,len_x)
  ax3.legend(loc='upper right',fontsize=14)
  ax3.invert_yaxis()
  ax3.set_yticks([])  
  return (fig,ax1,ax2,ax3)