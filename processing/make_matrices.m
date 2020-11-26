function [feature_mat,stats_features,feature_struct, raw_mat, raw_mat_g] = make_matrices(data, fsppg, nA,nB,nM,p_ok_pulses)
    
    % INPUT
    %   data -> 2xM half-normalized matrix (1 -> ppg signal (noramlized), 2 -> abp signal (non-normalized))
    %   fsppg -> sampling frequency
    %   nA -> locations of the systolic peaks in the ppg signal
    %   nB -> starting locations of the ppg pulses
    %   filename -> name of the file the pulses were extracted from
    %   make1 -> boolean, flag for saving the feature matrix
    %   make2 -> boolean, flag for saving the raw matrix
    %   testDir -> directory where to save all raw signals and peaks/valleys
    % OUTPUT
    %   feature_mat -> N*15 matrix of features extracted from ppg pulses
    %   raw_mat -> N*(fsppg * 2 + 3) matrix containing centered raw values
    %       (normalized) of ppg pulses + systolic and distolic blood pressure values
    %   feature_struct -> struc with all the ppg features ()
    %   stats_features -> struct with mean(feature_struct) and std(feature_struct) 

    
    % Original from:
    % Slapničar, G.; Mlakar, N.; Luštrek, M. Blood Pressure Estimation from Photoplethysmogram UsingSpectro-Temporal Deep Neural Network. 19, 3420. doi:10.3390/s19153420.
    % Source:
    % https://github.com/gslapnicar/bp-estimation-mimic3/blob/master/cleaning_scripts/
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Modified by Nicolas Aguirre    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Default values
        if nargin<4
            error('Not enough input arguments');
        end
        % plots 1 and 2 (only for visualization during testing)
        %Plot original pulse
        p1 = false;
        
        %Plot the points if the pulse was accepted
        p2 = false;

    %% Main loop

    % margin of the abp section and initialization of feature and raw matrices
    margin = 20;
    feature_mat = [];
    stats_features = [];
    raw_mat_g = [];
    %Extreme values of SBP and DBP
    dbp_extreme_low = 50;
    sbp_extreme_low = 80;
    dbp_extreme_high = 120;
    sbp_extreme_high = 200;
    
    %Extreme values for pulse lenght
    max_pulse_lenght = 1.5 * fsppg;
    min_pulse_lenght = 0.5 * fsppg;
    n_pulses_ub = length(data(1,:))/min_pulse_lenght;
    n_pulses_lb = length(data(1,:))/max_pulse_lenght;
    
    %Percentage of good pulses in the signals
    %p_ok_pulses = 0.8;
    
    % first and second dervatives of the ppg signal, for features extraction later on
    vpg = diff(data(1,:));
    vpg = [vpg(1), vpg];
    apg = diff(vpg);
    apg = [apg(1), apg];
    
    q_pulses = (size(nA,2)-1);
    if q_pulses<n_pulses_lb || q_pulses>n_pulses_ub            
        %If problem, return empty. 
        stats_features = [];
        feature_struct = [];
        raw_mat = [];
        return;
    end
    control_vector = true(1,q_pulses);
    
    %Pre-Allocation
    feature_struct = func_create_struct_fe_PPG(q_pulses);
    raw_mat = zeros(q_pulses,2*fsppg+1);    
    
    for i = 1:q_pulses       
        %% Check for anomalies
        if(isnan(nA(i)) || isnan(nB(i)))
            continue
        end
        % check if the given pulse indices point to NaN data
        if(isnan(data(1,nA(i))) || isnan(data(1,nB(i))))
            continue
        end
        if(isnan(data(2,nA(i))) || isnan(data(2,nB(i))))
            continue
        end
        
        %% Check for any abnormal spikes in Systolic peaks (PPG)
        
        % throw away the whole recording if this happens, since we dont have an idea what the correct signal is anymore
        % if the difference in peaks is (> 20 mmHg) in under 10 seconds
        if i ~= 1 && (data(1,nA(i+1)) - data(1,nA(i))) > 20 && (nB(i+1) - nB(i)) > 10*fsppg
            feature_mat = [];
            raw_mat = [];
            return; 
        end
        
        %% Extract PPG, VPG, APG and ABP sections
        ppg_section = data(1, nB(i):nB(i+1));
        vpg_section = vpg(1, nB(i):nB(i+1));
        apg_section = apg(1, nB(i):nB(i+1));
        if size(ppg_section,2) > 2*fsppg
            %fprintf('%d PPG section is too large: make it max 2* sampling frequency\n', i);
            continue;
        end
        %Skewness
        skew = skewness(ppg_section);
        if skew < 0
            continue;
        end
        %Index beat�s
        left_margin = max(1,nB(i)-margin);
        right_margin = min(size(data,2),nB(i+1)+margin);
        abp_section = data(2, left_margin:right_margin);

        % size equals to the sampling frequency*2 + 1 (helps to be odd)
        raw_new_g = zeros(1,2*fsppg+1 +4);
        % center the raw signal
        padding = ceil((size(raw_new_g,2)-2 - size(ppg_section,2))/2); %Q zeros a cada costado de la señal centrada
        raw_new_g(padding+1:padding+size(ppg_section,2)) = ppg_section;% 
        target_features = gasper_all_raw_cycles(abp_section);
        if ~any(target_features < 0)
            %raw_new_g(end-3:end) = gasper_all_raw_cycles(abp_section);
            raw_new_g(end-3:end) = target_features;
            raw_mat_g = [raw_mat_g; raw_new_g];
        end

        %%Plot original pulse
        if(p1)
            figure(i+50)
            subplot(1,2,1)
            plot(ppg_section)
            title('PPG')
            subplot(1,2,2)
            plot(abp_section)
            title('ABP')
        end
        
        %% Extract the features of the current pulse
        
        % evaluate the pulse and calculate the point of diastolic slope
        % returns -1 if the diastolic slope doesnt exsist
        [dia_start, dia_peak] = check_pulse(ppg_section,nA(i)-nB(i)+1);
        if dia_peak == -1 || dia_start == -1
            %fprintf('no diastolic peak/ledge\n')
            continue
        end
        
        % first derivative features
        [w,y,z] = vpg_features(vpg_section, fsppg);
        if w == -1
            %fprintf('a pombear\n')
            continue
        end
        
        % second derivative features
        [a,b,c,d,e] = apg_features(apg_section);
        if a == -1
            %fprintf('a pombel\n')
            continue
        end
        
        % check the data for any anomalies that have gotten past the checks
        % so far and join the extracted features in a matrix, alogn with some addtional ones
        sys_peak = nA(i)-nB(i)+1;
        sys_idx = nA(i);

        
        %Delineator indx
        left = nB(i);
        right = nB(i+1);
        DicroN = nM((nM>=left) & (nM<=right) & (sys_idx<nM));
        if isempty(DicroN)
            continue
        end

        [features,struct] = make_features(ppg_section, vpg_section, apg_section, abp_section, fsppg, 1, sys_peak, dia_peak, dia_start, w,y,z,a,b,c,d,e);
        if isempty(features)
            continue 
        end
        %Index
        feature_struct(i).left_margin = left_margin;
        feature_struct(i).right_margin = right_margin;
        feature_struct(i).sys_peak = sys_peak;
        feature_struct(i).dia_start = dia_start;
        feature_struct(i).dia_peak = dia_peak;
        %Index added by Aguirre
        feature_struct(i).left = left;
        feature_struct(i).right = right;
        feature_struct(i).sys_idx = sys_idx;
        feature_struct(i).dicN_idx = DicroN;
        feature_struct(i).skew = skew; % Skew
        
        feature_struct(i).w = w;
        feature_struct(i).y = y;
        feature_struct(i).z = z;
        feature_struct(i).a = a;
        feature_struct(i).b = b;
        feature_struct(i).c = c;
        feature_struct(i).d = d;
        feature_struct(i).e = e;
        %Ratios
        feature_struct(i).bd = struct.bd;
        feature_struct(i).scoo = struct.scoo;
        feature_struct(i).sc = struct.sc;
        feature_struct(i).bcda = struct.bcda;
        feature_struct(i).cw = struct.cw;
        feature_struct(i).bcdea = struct.bcdea;
        feature_struct(i).sdoo = struct.sdoo;
        feature_struct(i).cs = struct.cs;
        %Time
        feature_struct(i).tc = struct.tc;   % complete time
        feature_struct(i).ts= struct.ts;    % time to sys peak
        feature_struct(i).td = struct.td;   % time from sys peak till the end
        feature_struct(i).tod = struct.tod;  % time to dia-peak
        feature_struct(i).tnt = struct.tnt;  % time from sys peak to dia peak
        feature_struct(i).ttn = struct.ttn;  % time from dia peak till the end
        %Areas
        feature_struct(i).auc_sys = struct.auc_sys;
        feature_struct(i).aac_sys = struct.aac_sys;
        feature_struct(i).auc_dia = struct.auc_dia;
        feature_struct(i).aac_dia = struct.aac_dia;
        %Target
        feature_struct(i).mean_sbp = struct.mean_sbp;
        feature_struct(i).mean_dbp = struct.mean_dbp;
        feature_struct(i).sbp_class = struct.sbp_class; 
        feature_struct(i).dbp_class = struct.dbp_class;

        % add a timestamp (starting sample number / sampling frequency to get the time in seconds) to the features vector
        features = [nB(i)/fsppg,features];
        % add feature vector to the feature matrix
        feature_mat = [feature_mat; features];

        % make raw_signal matrix (ppg)
        % size equals to the sampling frequency*2 + 1 (helps to be odd)
        %raw_new = zeros(1,2*fsppg+1 +4);
        raw_new = zeros(1,2*fsppg+1);
        % center the raw signal
        %padding = ceil((size(raw_new,2)-2 - size(ppg_section,2))/2);
        padding = ceil((size(raw_new,2) - size(ppg_section,2))/2);
        raw_new(padding+1:padding+size(ppg_section,2)) = ppg_section;
        %raw_new(end-3:end) = [features(end-3),features(end-2), features(end-1), features(end)];
        raw_mat(i,:) = raw_new;      

        %% Plot the points if the pulse was accepted
        if(p2)
            figure(i)
            subplot(1,3,1)
            hold on;
            plot(ppg_section)
            scatter(nA(i)-nB(i)+1,data(1,nA(i)))
            scatter(dia_peak,ppg_section(1,dia_peak))
            scatter(dia_start,ppg_section(1,dia_start))
            title('PPG')
            hold off;
            subplot(1,3,2)
            plot(vpg_section)
            title('VPG (1st derivative)')
            hold on;
            scatter(w,vpg_section(w))
            scatter(y,vpg_section(y))
            scatter(z,vpg_section(z))
            subplot(1,3,3)
            plot(apg_section)
            hold on;
            scatter(a,apg_section(a))
            scatter(b,apg_section(b))
            scatter(c,apg_section(c))
            scatter(d,apg_section(d))
            scatter(e,apg_section(e))
            title('APG (2nd derivative)')
        end
    end
    %% Eliminate NaN Pulses
    for i_Cycle = 1:q_pulses
        for fn = fieldnames(feature_struct)'
            fieldcontent = feature_struct(i_Cycle).(fn{1});
            if isnan(fieldcontent)
                control_vector(1,i_Cycle)=false;
            end
        end
    end
    feature_struct = feature_struct(control_vector);
    raw_mat = raw_mat(control_vector,:);
    
    % Quantity of "good" pulses control
    % Percentage of good pulses in the signals
    if size((feature_struct),2) < q_pulses*p_ok_pulses
        feature_struct = [];
        raw_mat=[];
        return;
    end

    feature_struct_pivot = struct2table(feature_struct,'AsArray',true);

    for fn = feature_struct_pivot.Properties.VariableNames
        %don't care this
        if (string(fn{1}) =='left_margin' || string(fn{1}) =='right_margin' ||...
                string(fn{1}) =='left' || string(fn{1}) =='right' ||...
                string(fn{1}) =='sys_idx' || string(fn{1}) =='dicN_idx') %#ok<*BDSCA>
            continue
        end

        if (string(fn{1}) ~='SBP' && string(fn{1}) ~='DBP')
            values = feature_struct_pivot.(fn{1});
            mean_v = mean(values);
            std_v = std(values);
            ub = mean_v+3*std_v;
            lb = mean_v-3*std_v;
            mean_v = mean(values(values<=ub & values>=lb));
            std_v = std(values(values<=ub & values>=lb));

        else
            % calculate the average of peaks in ABP + checks
            if (fn{1} =='SBP')
                sbp_peaks = feature_struct_pivot.(fn{1});
                max_sbp = max(sbp_peaks);
                mean_v = mean(sbp_peaks);
                std_v = std(sbp_peaks);
                ub = mean_v+3*std_v;
                lb = mean_v-3*std_v;
                mean_v = mean(...
                    sbp_peaks((max_sbp - sbp_peaks) < 30 &... %Max Diff
                    sbp_peaks > sbp_extreme_low & sbp_peaks < sbp_extreme_high &... %Range Values
                    sbp_peaks<ub & sbp_peaks>lb));  %Stat conf.
                std_v = std(...
                    sbp_peaks((max_sbp - sbp_peaks) < 30 &... %Max Diff
                    sbp_peaks > sbp_extreme_low & sbp_peaks < sbp_extreme_high &... %Range Values
                    sbp_peaks<ub & sbp_peaks>lb));  %Stat conf.
            end
            % calculate the average dbp + checks        
            if (fn{1} =='DBP')
                dbp_values = feature_struct_pivot.(fn{1});
                min_dbp = min(dbp_values(dbp_values > dbp_extreme_low & dbp_values < dbp_extreme_high));
                mean_v = mean(dbp_values);
                std_v = std(dbp_values);
                ub = mean_v+3*std_v;
                lb = mean_v-3*std_v;
                mean_v = mean(...
                    dbp_values((dbp_values - min_dbp) < 7 &...  %Max diff
                    dbp_values > dbp_extreme_low & dbp_values < dbp_extreme_high &... %Range Values;
                    dbp_values<ub & dbp_values>lb));    %Stat conf.
                std_v = std(...
                    dbp_values((dbp_values - min_dbp) < 7 &...  %Max diff
                    dbp_values > dbp_extreme_low & dbp_values < dbp_extreme_high &... %Range Values;
                    dbp_values<ub & dbp_values>lb));    %Stat conf.
            end
        end

        %Control
        if isempty(mean_v) || isnan(mean_v) || isempty(std_v) || isnan(std_v)
            %If problem, return empty. 
            stats_features = [];
            feature_struct = [];
            raw_mat = [];
            return;
        end
    %Structure
    stats_features.mean.(fn{1})=mean_v;
    stats_features.std.(fn{1})=std_v;
    end

end