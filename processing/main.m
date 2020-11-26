clc;
close all;
clear all;


%%PLOTS
show_flat_lines = false;
show_flat_peaks = false; 
show_delineator = false;
plot_pan_tompkins = false;
plot_PTT = false;
plot_mean_pulse = false;
plot_final = false;
save_processing = false;

%% Folders
if ispc
    %Windows (trabajando siempre sobre el google drive) 
    %Processing Folder
    addpath('ADJUST TO YOUR PATH\processing')
    output_path = 'ADJUST TO YOUR PATH\datasets\ABP_PPG';
    %Filtered Dataframe
    addpath('ADJUST TO YOUR PATH\\df_filtered')
    cd('ADJUST TO YOUR PATH\\processing')
else
    % Linux
    % Processing Folder
    addpath('ADJUST TO YOUR PATH/processing')
    output_path ='ADJUST TO YOUR PATH/datasets/ABP_PPG/'; 
    % Filtered Dataframe Folder
    addpath('ADJUST TO YOUR PATH/df_filtered/ABP_PPG')
    % MIMIC-III WAVEFORM DB MATCHED SUBSET
    addpath('ADJUST TO YOUR PATH/mimic3wdb/matched')
    % WFDB-Library Folder
    addpath('ADJUST TO YOUR PATH/mcode')
end

%Pandas Dataframe (from Pre-Processing)
df_filtered = readtable('df_filtered_signals_icu_hadm_sex_age.csv');
wfdbloadlib
[~,config] = wfdbloadlib;
indx_signal_char =  regexp(df_filtered.idx_signal, {'\[\d\]'});
indx_signal_char = cellfun(@(x) x+1,indx_signal_char,'un',0);

%% Parameters
load 'parametros.mat' % --> Input Channels
ppg_c = find(strcmp('PLETH',cellstr(channels_selected)));
abp_c = find(strcmp('ABP',cellstr(channels_selected)));
ecg_c = find(strcmp('II',cellstr(channels_selected)));
rbp_c = find(strcmp('RBP',cellstr(channels_selected)));
fbp_c = find(strcmp('FBP',cellstr(channels_selected)));

%Include PTT Feature Extraction (FE)
PTT = false;
% Include Mean Pulse FE
mean_pulse_processing = true;

%% Script Start

q_channels_selected = size(channels_selected,1);
idx_signal = zeros(size(df_filtered.idx_signal,1),q_channels_selected);
% Channels indexs
for i = 1:size(idx_signal,1)
    for i_c = 1:q_channels_selected
        %'+1'adjust channel w.r.t. python
        idx_signal(i,i_c) = str2double(df_filtered.idx_signal{i}(indx_signal_char{i}(i_c)))+1; 
    end
end
df_filtered.idx_signal =  idx_signal;

% number of files
n_files = size(df_filtered,1);

% windows threshold (from Gašper Slapničar: https://github.com/gslapnicar/bp-estimation-mimic3)
w_flat = 15;    % flat lines window
w_peaks = 5;    % flat peaks window
w_fix = 15;     % flat join window ?

% thresholds (from Gašper Slapničar: https://github.com/gslapnicar/bp-estimation-mimic3)
t_peaks = 0.05; % percentage of tolerated flat peaks
t_flat = 0.05;  % percentage of tolerated flat lines

% Percentaje of Ok pulses in a segment
p_ok_pulses = 0.75;

%time to index
fs = 125; %sampling rate
len_signal = ventana*fs;
intervalo = intervalo * 60 * fs; % intervalo in min --> sec
i_global = signal_i*60*fs; % signal_i in min -->sec
ciclo = (intervalo+len_signal);
ciclo_total = ciclo *(q_signal-1)+len_signal;

% windows shifted by 'ciclo' o 'len_signal' o scale of it
% if the current segment not satisfice the quality
shifted_window = fs*60; %(1 min)

%Relative index to i_global for every i_q_signal
i_idx_tot_sgn = zeros(q_signal,1);
f_idx_tot_sgn = zeros(q_signal,1);
i_idx_proc_sgn = zeros(q_signal,1);
f_idx_proc_sgn = zeros(q_signal,1);
for i_q = 1:q_signal
    % Inicio para la señal total (separados un ciclo)
    i_idx_tot_sgn(i_q) = (i_q-1)*ciclo;
    % Inicio para los datos a procesar
    i_idx_proc_sgn(i_q) = 1+(i_q-1)*len_signal; 
    % Fin para la señal total (separados 'len_signal' del incio)
    f_idx_tot_sgn(i_q) = i_idx_tot_sgn(i_q)+len_signal;
    % Fin para los datos procesados (separados 'len_signal' del incio)
    f_idx_proc_sgn(i_q) = i_idx_proc_sgn(i_q)+len_signal;
end

%Accumulated q_signal array (Channels ,total_length)
acumm_signal = zeros(q_channels_selected,len_signal*q_signal);

disp(strcat('Start processing:'))
q_ok = 0;
%Previous proccesing ...
processing_results = func_processing_results(n_files,channels_selected);
filename_completed_processing = strcat(output_path,'completed', '.mat');
if isfile(filename_completed_processing)
    past_processed = load(filename_completed_processing);
    n_processed =size(past_processed.completed,2);
    processing_results(1:n_processed)=past_processed.completed;
    q_ok = sum([processing_results(1:n_processed).status]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -------------------------- Main Loop ----------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:n_files
    %% Load signal + info
    process_time = 0;
    row = df_filtered(i,:);
    %fprintf('Processing subrecord: %s\r\n', row.subrecord{1})
    pxx = row.pxx{1};
    pxxxxxx = row.pxxxxxx{1};
    subrecord = row.subrecord{1};
    canales = strlength(row.signals{1});
    file_name = subrecord;% num2str(row.Var1); %Idx from df_full_column
    
    % Chek if was already processed
    i_process = find(strcmp({processing_results(:).id},file_name)==1);
    if ~isempty(i_process)
        process_status = processing_results(i_process).status;
        if process_status==0
            %fprintf('Already rejected: %s\r\n',file_name)
            continue
        end
        if process_status==1
            %fprintf('Already processed: %s\r\n', file_name)           
            continue 
        end
    end    
    %Load depending on OS
    if ispc
        signal_path=[config.WFDB_PATH pxx filesep pxxxxxx filesep ];
        cd(signal_path);
        signal = rdsamp(subrecord);
    else 
        signal_path=[filesep pxx filesep pxxxxxx filesep subrecord];
        signal = rdsamp(signal_path);
    end
    
    %Only the channels selected in "idx_signal"
    signal = signal(:,row.idx_signal);
    %[T,channels] -> [Channels,Time]
    signal = permute(signal,[2,1]);
    % Fail counts
    count_cycles = 0;
    count_f = 0;
    count_p = 0;
    count_fe = 0;
    count_fe_ppg = 0;
    count_fe_abp = 0;
    %
    total_signal_len = row.length_ms;
    i_global = signal_i*60*fs; %Global initial time for each signal
    t_final = i_global + ciclo_total; %Reset el t_final
    flag_feature = 1; % Initialice the flag 0=complet, 1= not complet
    i_q_signal = 1; % count of good q_signal to try F.Extraction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % -------------------------- While loop-------------------------------%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    while (i_q_signal <= q_signal) && (t_final < total_signal_len) && (flag_feature==1)
        %% Detect flat lines in the signal
        %This is the i_q_signal element to evaluate
        data = signal(:,i_global+i_idx_tot_sgn(i_q_signal):i_global+f_idx_tot_sgn(i_q_signal));
        [per_mat] = flat_lines(data, w_flat, false,show_flat_lines);
        count_cycles = count_cycles+1;
        if any(per_mat > t_flat) % Bad i_q_signal
            %Shift
            i_global = i_global + shifted_window*i_q_signal;
            t_final = i_global + ciclo_total;
            count_f = count_f + i_q_signal;
            i_q_signal = 1;
        else 
        %% Detect flat peaks in the signal
            sum_flat = 0;
            if (~isempty(ppg_c))
                [~, ppg_peaks] = findpeaks(data(ppg_c,:)); [~, ppg_valleys] = findpeaks(-1 * data(ppg_c,:));
                a = flat_peaks_general(data(ppg_c,:),ppg_peaks,ppg_valleys, t_peaks, w_peaks, show_flat_peaks);
                sum_flat = sum_flat + a;
            end
            if (~isempty(abp_c))
                [~, abp_peaks] = findpeaks(data(abp_c,:)); [~, abp_valleys] = findpeaks(-1 * data(abp_c,:));
                b = flat_peaks_general(data(abp_c,:),abp_peaks,abp_valleys, t_peaks, w_peaks, show_flat_peaks);
                sum_flat = sum_flat + b;
            end
            
            if(sum_flat)
                %Shift
                i_global = i_global + shifted_window*i_q_signal;
                t_final = i_global + ciclo_total;
                count_p = count_p + i_q_signal;
                i_q_signal = 1;
            else
                %At this points we are able to go to F.Extraction
                acumm_signal(:,i_idx_proc_sgn(i_q_signal):f_idx_proc_sgn(i_q_signal)) = data;
                i_q_signal = i_q_signal+1;
            end 
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % -------- Feature Extraction main loop-----------------------%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if (i_q_signal > q_signal)
            count_cycles = count_cycles - q_signal;
            % "Good" feature extraction consideration a priori. if any
            % feature can't be extracted, flag_feature==1, and the "while"
            % not gona be stoped.
            flag_feature = 0;
            for i_q_signal_fe = 1:q_signal
                if flag_feature == 0 % is 0 during feature extraction
                    %% PPG FEATURES
                    if (~isempty(abp_c) && ~isempty(ppg_c))
                        %data = filter_signal(acumm_signal, fsppg); 
                        raw_ppg_abp = acumm_signal([ppg_c,abp_c],i_idx_proc_sgn(i_q_signal_fe):f_idx_proc_sgn(i_q_signal_fe));
                        % bandpass and hampel filters
                        data = filter_signal(raw_ppg_abp, fs);
                        temp_ppg_c=1; %Temporal index channel to PPG
                        temp_abp_c=2; %Temporal index channel to ABP
                        
                        % Join relevant valleys to remove flat lines
                        % using the Pulse waveform deliniator (quick)
                        % onsetp,peak,dicron
                        [nB2,nA2,nM2] = delineator(data(temp_ppg_c,:), fs);
                        new_data = flat(data, w_fix, nB2, 'ppg', false);        %ppg - 1
                        [nB3,nA3,nM3] = delineator(new_data(temp_abp_c,:), fs);
                        new_data = flat(new_data, w_fix, nB3, 'abp', false);    %abp - 2
                        foot=[];SysPeak=[];DicroN=[];
                        foot.PPG = nB2;SysPeak.PPG = nA2;DicroN.PPG=nM2;
                        foot.ABP = nB3;SysPeak.ABP = nA3;DicroN.ABP=nM3;
                        %Plot
                        if(show_delineator)
                            func_plot_delineator(data,foot,SysPeak,DicroN);
                        end
                        % MinMax [0-1] Only PPG
                        norm_data = normalize_signal(new_data(temp_ppg_c,:)); 
                        norm_data = [norm_data; new_data(temp_abp_c,:)]; %Add ABP w/o normalize
                        % Feature Extraction - make a feature matrix and a matrix of raw values
                        [feat,stats_features_ppg, features_ppg, raw] = make_matrices(norm_data, fs, nA2, nB2,nM2,p_ok_pulses);
                        %If is empty or not enough pulses good, quit and search
                        %in next step
                        if(isempty(feat) || size(feat,1) < ventana*0.5 || isempty(raw) || size(raw,1) < ventana*0.5)
                            struct_features = [];
                            struct_stats_features= [];
                            flag_feature = 1;
                            %Indx update
                            i_global = i_global + shifted_window*i_q_signal_fe;
                            t_final = i_global + ciclo_total;
                            %Update fail count
                            count_cycles = count_cycles + i_q_signal_fe;
                            count_fe = count_fe + i_q_signal_fe;
                            count_fe_ppg = count_fe_ppg + i_q_signal_fe;
                            i_q_signal = 1;
                            continue
                        end
                        struct_features(i_q_signal_fe).ppg = features_ppg;
                        struct_stats_features(i_q_signal_fe).ppg = stats_features_ppg;
                        struct_delineator(i_q_signal_fe).PPG.foot = foot.PPG;
                        struct_delineator(i_q_signal_fe).PPG.SysPeak = SysPeak.PPG;
                        struct_delineator(i_q_signal_fe).PPG.DicroN = DicroN.PPG;
                    end
                    %% ABP FEATURES
                    if (~isempty(abp_c))
                        abp = acumm_signal(abp_c,i_idx_proc_sgn(i_q_signal_fe):f_idx_proc_sgn(i_q_signal_fe));
                        %foot,peakp,DicroN
                        [nB4,nA4,nM4] = delineator(abp,fs);
                        foot=[];SysPeak=[];DicroN=[];
                        foot.ABP = nB4;SysPeak.ABP = nA4;DicroN.ABP=nM4;
                        %Plot
                        %if(show_delineator)
                        %    func_plot_delineator(abp,foot,SysPeak,DicroN);
                        %end
                        %Features Extraction ABP
                        [stats_features_abp, features_abp,raw_signals_abp] = func_features_ABP('signal',abp,foot.ABP,SysPeak.ABP,DicroN.ABP,fs,p_ok_pulses);

                        if(isempty(features_abp) || size(features_abp,2) < ventana*0.5 || isempty(raw_signals_abp))
                            struct_features = [];
                            struct_stats_features = [];
                            flag_feature = 1;
                            %Indx update
                            i_global = i_global + shifted_window*i_q_signal_fe;
                            t_final = i_global + ciclo_total;
                            %Update fail count
                            count_cycles = count_cycles + i_q_signal_fe;
                            count_fe = count_fe + i_q_signal_fe;
                            count_fe_abp = count_fe_abp + i_q_signal_fe;
                            i_q_signal = 1;
                            continue
                        end
                        if(show_delineator)
                            func_plot_delineator(abp,foot,SysPeak,DicroN);
                        end
                        struct_features(i_q_signal_fe).abp = features_abp;
                        struct_features(i_q_signal_fe).beats_abp = raw_signals_abp;
                        struct_stats_features(i_q_signal_fe).abp = stats_features_abp;
                        struct_delineator(i_q_signal_fe).ABP.foot = foot.ABP;
                        struct_delineator(i_q_signal_fe).ABP.SysPeak = SysPeak.ABP;
                        struct_delineator(i_q_signal_fe).ABP.DicroN = DicroN.ABP;
                        
                        if mean_pulse_processing
                            [features_mean_pulse, mean_pulse_abp] = func_feature_ABP_mean_pulse(raw_signals_abp,stats_features_abp.mean,fs,plot_mean_pulse);                        
                            struct_features(i_q_signal_fe).mean_pulse_abp = mean_pulse_abp;
                            struct_stats_features(i_q_signal_fe).maen_abp = features_mean_pulse;
                        end
                        
                    end
                    
                    %% PTT ECG - ABP - PPG
                    if PTT == true
                        ecg = acumm_signal(ecg_c,i_idx_proc_sgn(i_q_signal_fe):f_idx_proc_sgn(i_q_signal_fe));
                        abp = acumm_signal(abp_c,i_idx_proc_sgn(i_q_signal_fe):f_idx_proc_sgn(i_q_signal_fe));
                        ppg = acumm_signal(ppg_c,i_idx_proc_sgn(i_q_signal_fe):f_idx_proc_sgn(i_q_signal_fe));
                        %Pan-Tompkin
                        [qrs_amp_raw,qrs_i_raw,delay] = pan_tompkin(ecg,fs,plot_pan_tompkins);
                        abp_foot = [features_abp.left]; %Only for those ABP features extracted
                        ppg_foot = nB2;
                        method = 1; %wrt ECG
                        [stats_PTT, feature_PTT, ~] = func_PTT(1,ecg,abp,ppg,qrs_i_raw,abp_foot,ppg_foot,fs,plot_PTT); %PTT in ms
                        struct_features(i_q_signal_fe).PTT = feature_PTT;
                        struct_stats_features(i_q_signal_fe).PTT = stats_PTT;
                        if(isempty(feature_PTT) || size(feature_PTT,2) < ventana*0.5)
                            struct_features = [];
                            struct_stats_features = [];
                            flag_feature = 1;
                            %Indx update
                            i_global = i_global + shifted_window*i_q_signal_fe;
                            t_final = i_global + ciclo_total;
                            %Update fail count
                            count_cycles = count_cycles + i_q_signal_fe;
                            count_fe = count_fe + i_q_signal_fe;
                            %count_fe_abp = count_fe_abp + i_q_signal_fe;
                            i_q_signal = 1;                
                            continue
                        end
                        struct_delineator(i_q_signal_fe).ECG = qrs_i_raw;
                    end
                    
                end
            end
        end
        %% Saving signal and feature
        if flag_feature==0
            for i_signal = 1:q_signal
                
                signal_processing(i_signal).signal=acumm_signal(:,i_idx_proc_sgn(i_signal):f_idx_proc_sgn(i_signal));
                signal_processing(i_signal).index = [i_global+i_idx_tot_sgn(i_signal),i_global+f_idx_tot_sgn(i_signal)];
                signal_processing(i_signal).struct_features = struct_features(i_signal);
                signal_processing(i_signal).struct_stats_features = struct_stats_features(i_signal);
                signal_processing(i_signal).struct_delineator = struct_delineator(i_signal);
            end
            
            if (plot_final)
                figure();
                for i_chn = 1:q_channels_selected
                    subplot(q_channels_selected,1,i_chn)
                    x_axes = i_global+i_idx_tot_sgn(1):i_global+f_idx_tot_sgn(end);
                    y_val = signal(i_chn,i_global+i_idx_tot_sgn(1):i_global+f_idx_tot_sgn(end));
                    plot(x_axes, y_val, 'k','LineWidth',0.1);
                    hold on
                    for i_signal = 1:q_signal
                        plot(i_global+i_idx_tot_sgn(i_signal):i_global+f_idx_tot_sgn(i_signal),signal(i_chn,i_global+i_idx_tot_sgn(i_signal):i_global+f_idx_tot_sgn(i_signal)),'b')
                    end
                    xlabel('Time (s)', 'FontSize', 20)
                    title('Processed segments','FontSize', 20)
                end
                
            end

            if save_processing == true
                filename_signal_processing = strcat(output_path,file_name, '.mat');
                save(filename_signal_processing,'signal_processing');
            end
            status = 1;
            clear signal_processing;
        end 
    end
    
    %% Fail Saving .mat
    if flag_feature==1
        status = 0;
        %filename_signal_processing = strcat(output_path,'fail/',file_name, '.mat');
        %save(filename_signal_processing,'file_name');
    end
    
    %% Update Process status
    q_ok = q_ok + status;
    if (mod(i,15)==0)
        fprintf('Completed %i/%i\r\n',i,n_files);
        fprintf('Acepted %i\r\n',q_ok);
        fprintf('Rejected %i\r\n',i-q_ok);
    end
    
    if count_cycles - count_f - count_p - count_fe~=0
        error('Error in count_cycles');
    end
    
    processing_results(i).id = file_name;
    processing_results(i).status = status;
    processing_results(i).count_cycles = count_cycles;
    processing_results(i).count_f = count_f;
    processing_results(i).count_p = count_p;
    processing_results(i).count_fe = count_fe;
    processing_results(i).count_fe_ppg = count_fe_ppg;
    processing_results(i).count_fe_abp = count_fe_abp;

    completed = processing_results(1:i);
    
    if save_processing == true
        save(filename_completed_processing,'completed');
    end
end

fprintf('Finished \r\n');