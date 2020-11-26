function [stats, features,raw_signals] = func_features_ABP(method,abp,foot,syst,DicroN,fs,p_ok_pulses)
%Feature extraction of ABP signals based on Baed on SALVI pag:140-141
%{
Inputs:
-method: 'signal' is for feature extraction based on continuos pulses.
'mean_pulse' is for feature extraction on mean_pulse
-abp: abp input data
-foot: index of each pulses start point in abp
-syst: index of each systolic peak in abp
-DicroN: index of each DicroticNotch point in abp
-fs: Sampling frecuency

Outputs:
-stats: mean and stdv of each feature
-features: table containing features of each valid pulse
-raw_signals: abp separated in pulses. size = (N_valid pulses,2*fs). Pulses
are not centered, they start at index 1.
%}

%Initilize variables
features = [];
raw_signals = [];
stats = [];

%Extreme values for pulse lenght
max_pulse_lenght = 1.5 * fs;
min_pulse_lenght = 0.5 * fs;

%Extreme values of SBP and DBP
dbp_extreme_low =50;
sbp_extreme_low = 80;

dbp_extreme_high =120;
sbp_extreme_high =200;
%Extreme values of pulses in signal
n_pulses_ub = length(abp)/min_pulse_lenght;
n_pulses_lb = length(abp)/max_pulse_lenght;

%Percentage of good pulses in the signals
%p_ok_pulses = 0.5;

%StandDev to consider part of the mean
sd_lim = 2;

    switch method
        case 'signal'         %Based on SALVI pag:140-141
            q_pulses = (size(syst,2)-1);
            control_vector = true(1,q_pulses); 

            %Many/few pulse for the abp signal--> fail in delineator / signal quality
            if q_pulses<n_pulses_lb || q_pulses>n_pulses_ub            
                stats = [];
                features = [];
                raw_signals = [];
                return;
            end
            
            %Pre-Allocation
            features = func_create_struct_fe_ABP(q_pulses);
            raw_signals = zeros(q_pulses,2*fs+1);
            
            for i_Cycle = 1:q_pulses
                %Index beats
                left = foot(i_Cycle);
                right = foot(i_Cycle+1);
                if left+max_pulse_lenght<right || left + min_pulse_lenght>right
                    % avoid wrong large || small pulse lenght
                    continue
                end            
                %% Indx
                sys_idx = syst((syst>=left) & (syst<=right));
                dicN_idx = DicroN((DicroN>=left) & (DicroN<=right) & (sys_idx<DicroN));
                %% Phase BP
                bp_section = abp(left:right);
                sys_phase = abp(left:dicN_idx);
                dias_phase = abp(dicN_idx:right);
                skew = skewness(bp_section);
                if skew < 0
                    % avoid wrong large || small pulse lenght
                    continue
                end
                %% Time Features
                % Time to Systolic Peak
                iTTSP = (sys_idx - left);
                TTSP = iTTSP/fs;
                % Left ventricular ejection time
                iLVET = (dicN_idx - left);
                LVET = iLVET/fs;   
                % Diastolic time - Duration of the diastolic phase of the cardiac cycle
                iDT = (right - dicN_idx);
                DT = iDT/fs;
                % Heart period
                iHP = (right-left);
                HP = (iHP)/fs;
                % Diastolic time fraction
                DTF = DT / HP;
                %% Pressure Features
                % systolic blood pressure
                SBP =abp(sys_idx);
                % Diastolic blood pressure
                DBP = abp(right); %Al comienzo o al final?
                %Control SBP-DBP values
                control_flag = control_SBP_DBP(SBP,DBP,...
                    dbp_extreme_low,sbp_extreme_low,...
                    dbp_extreme_high,sbp_extreme_high);
                if control_flag ==1
                    stats = []; 
                    features = [];
                    raw_signals = [];
                    return;
                end
                % End-systolic blood pressure / Dicrotic Notch blood pressure
                ESBP = abp(dicN_idx);
                % Extreme values of Dicrotic Notch wrt SBP and DBP
                %if ESBP - 10 < DBP
                %    continue
                %end
                % Pulse Pressure
                PP = SBP - DBP;
                % Mean arterial pressure
                MAP = mean(bp_section);
                % Mean pulse pressure
                MPP = MAP - DBP;
                % Mean systolic blood pressure
                MSBP = mean(sys_phase);
                % Mean diastolic blood pressure
                MDBP = mean(dias_phase);
                %% Wrong, Not fiable, avoid use as feature
                % Travel time of the reflected wave
                Ti = FUN_ANAL_AIx_modificado(bp_section,1,iHP,SBP,iTTSP,DBP,1,0);
                if isempty(Ti)
                    continue
                end
                % Blood pressure at inflection point % Not fiable
                Pi = bp_section(Ti); % Not fiable
                % Augmented pressure % Not fiable
                if Ti > sys_idx
                    AP = -(SBP - Pi);
                else
                    AP = (SBP - Pi);
                end
                % Augmentation index % Not fiable
                AIx = AP/PP;
                %% Form Factor
                FF = (MPP /PP); 
                %% AREA
                %Mean
                %SPTI = MSBP * LVET; % Systolic pressure�time index (tension�time index)
                %DPTI = MDBP * DT;   % Diastolic pressure�time index
                %SEVR = DPTI / SPTI ;% Subendocardial viability ratio
                % Integrando
                % is possible also to rest foot bp value
                SPTI = trapz(sys_phase);    % Systolic pressure�time index (tension�time index)
                DPTI = trapz(dias_phase);   % Diastolic pressure�time index
                SEVR = DPTI / SPTI ;        % Subendocardial viability ratio
                %% To Struct
                %Indx
                features(i_Cycle).left = left;
                features(i_Cycle).right = right;
                features(i_Cycle).sys_idx = sys_idx;
                features(i_Cycle).dicN_idx = dicN_idx;
                features(i_Cycle).skew = skew;
                %Time 
                features(i_Cycle).iTTSP = iTTSP;
                features(i_Cycle).TTSP = TTSP;
                features(i_Cycle).iLVET = iLVET;
                features(i_Cycle).LVET = LVET;
                features(i_Cycle).iDT = iDT;
                features(i_Cycle).DT = DT;
                features(i_Cycle).iHP = iHP;
                features(i_Cycle).HP = HP;
                features(i_Cycle).DTF = DTF;
                %Pressure
                features(i_Cycle).SBP = SBP;
                features(i_Cycle).DBP = DBP;
                features(i_Cycle).ESBP = ESBP;
                features(i_Cycle).PP = PP;
                features(i_Cycle).Ti = Ti;
                features(i_Cycle).Pi = Pi;
                features(i_Cycle).AP = AP;
                features(i_Cycle).AIx = AIx;
                features(i_Cycle).MAP = MAP;
                features(i_Cycle).MPP = MPP;
                features(i_Cycle).MSBP = MSBP;
                features(i_Cycle).MDBP = MDBP;
                features(i_Cycle).FF = FF;
                %Area
                features(i_Cycle).SPTI = SPTI;
                features(i_Cycle).DPTI = DPTI ; 
                features(i_Cycle).SEVR = SEVR ;
                %features(i_Cycle).Ampli = Ampli;
                %features(i_Cycle).PPA = PPA;
                
                raw_new = zeros(1,2*fs+1);
                %padding = ceil((size(raw_new,2) - size(bp_section,2))/2); %Q zeros a cada costado de la señal centrada
                %raw_new(1,padding+1:padding+size(bp_section,2)) = bp_section;% 1 = BP
                %raw_signals = [raw_signals; raw_new];
                raw_new(1,1:size(bp_section,2)) = bp_section;% 1 = BP
                raw_signals(i_Cycle,:) = raw_new;
                
                %{
                %centered on systolic peak
                raw_new(x-(sys_idx-left):x+right) = bp_section
                %}
                %q=q+1;
            end
            %% Eliminate NaN Pulses
            for i_Cycle = 1:q_pulses
                for fn = fieldnames(features)'
                    fieldcontent = features(i_Cycle).(fn{1});
                    if isnan(fieldcontent)
                        control_vector(1,i_Cycle)=false;
                    end
                end
            end
            features = features(control_vector);
            raw_signals = raw_signals(control_vector,:);
            %Control of quantity of "good" pulses 
            if size((features),2) < q_pulses*p_ok_pulses
                features = [];
                raw_signals=[];
                stats=[];
                return;
            end

            features_pivot = struct2table(features,'AsArray',true);
            for fn = features_pivot.Properties.VariableNames
                %don't care this two.
                if (string(fn{1}) =='left' || string(fn{1}) =='right' ||...
                    string(fn{1}) == 'sys_idx' || string(fn{1}) == 'dicN_idx') %#ok<*BDSCA>
                    continue
                end
                
                if (string(fn{1}) ~='SBP' && string(fn{1}) ~='DBP')
                    values = features_pivot.(fn{1});
                    mean_v = mean(values);
                    std_v = std(values);
                    ub = mean_v+sd_lim*std_v;
                    lb = mean_v-sd_lim*std_v;
                    mean_v = mean(values(values<=ub & values>=lb));
                    std_v = std(values(values<=ub & values>=lb));
                else
                    % calculate the average of peaks in ABP + checks
                    if (fn{1} =='SBP')
                        sbp_peaks = features_pivot.(fn{1});
                        max_sbp = max(sbp_peaks);
                        mean_v = mean(sbp_peaks);
                        std_v = std(sbp_peaks);
                        ub = mean_v+sd_lim*std_v;
                        lb = mean_v-sd_lim*std_v;
                        mean_v = mean(...
                            sbp_peaks((max_sbp - sbp_peaks) < 30 &... %Max Diff
                            sbp_peaks<ub & sbp_peaks>lb));  %Stat conf.
                        std_v = std(...
                            sbp_peaks((max_sbp - sbp_peaks) < 30 &... %Max Diff
                            sbp_peaks<ub & sbp_peaks>lb));  %Stat conf.
                    end
                    % calculate the average DBP + checks        
                    if (fn{1} =='DBP')
                        dbp_values = features_pivot.(fn{1});
                        min_dbp = min(dbp_values);
                        mean_v = mean(dbp_values);
                        std_v = std(dbp_values);
                        ub = mean_v+sd_lim*std_v;
                        lb = mean_v-sd_lim*std_v;
                        mean_v = mean(...
                            dbp_values((dbp_values - min_dbp) < 7 &...  %Max diff
                            dbp_values<ub & dbp_values>lb));    %Stat conf.
                        std_v = std(...
                            dbp_values((dbp_values - min_dbp) < 7 &...  %Max diff
                            dbp_values<ub & dbp_values>lb));    %Stat conf.
                    end
                end

                %Control
                if isempty(mean_v) || isnan(mean_v) || isempty(std_v) || isnan(std_v)
                    %If problem, return empty. 
                    stats = [];
                    features = [];
                    raw_signals = [];
                    return;
                end
                
                %To Stats-STructure (Mean,Std)
                stats.mean.(fn{1})=mean_v;
                stats.std.(fn{1})=std_v;
            end
            %{
            %centered on systolic peak
            roll_over = stats.mean.iTTSP;
            raw_new = zeros(size(raw_signals,1),size(raw_signals,2));
            raw_new(:,1:(2*fs+1)-roll_over) = raw_signals(roll_over:end);
            raw_signals = raw_new;
            %}
            
        case 'mean_pulse'
            %Index beats
            left = 1;
            right = round(foot);

            sys_idx = round(syst);
            dicN_idx= round(DicroN);

            %Phase BP
            bp_section = abp(left:right);
            sys_phase = abp(left:dicN_idx);
            dias_phase = abp(dicN_idx:right);
            skew = skewness(bp_section);
            %% Time
            %Time to Systolic Peak
            iTTSP = (sys_idx - left);
            TTSP = iTTSP/fs;
            % Left ventricular ejection time
            iLVET = (dicN_idx - left);
            LVET = iLVET/fs;   
            % Diastolic time - Duration of the diastolic phase of the cardiac cycle
            iDT = (right - dicN_idx);
            DT = iDT/fs;
            % Heart period
            HP = right-left;
            % Diastolic time fraction
            DTF = DT / HP;
            %% Pressure Features
            % systolic blood pressure
            SBP =abp(sys_idx);
            % Diastolic blood pressure
            DBP = abp(left); %Al comienzo o al final? en el mean pulse tomo al comienzo.
            % End-systolic blood pressure
            ESBP = abp(dicN_idx); % Dicrotic Notch blood pressure
            % Pulse Pressure
            PP = SBP - DBP;
            % Travel time of the reflected wave
            Ti = FUN_ANAL_AIx_modificado(bp_section,left,right,SBP,sys_idx,DBP,1,0);
            % Blood pressure at inflection point
            Pi = bp_section(Ti);
            % Augmented pressure
            if Ti > sys_idx
                AP = -(SBP - Pi);
            else
                AP = (SBP - Pi);
            end
            % Augmentation index
            AIx = AP/PP;
            % Mean arterial pressure
            MAP = mean(bp_section);
            % Mean pulse pressure
            MPP = MAP - DBP;
            % Mean systolic blood pressure
            MSBP = mean(sys_phase);
            % Mean diastolic blood pressure
            MDBP = mean(dias_phase);

            %% AREA
            %Mean
            %SPTI = MSBP * LVET; % Systolic pressure�time index (tension�time index)
            %DPTI = MDBP * DT;   % Diastolic pressure�time index
            %SEVR = DPTI / SPTI ;% Subendocardial viability ratio
            % Integrando
            % is possible also to rest foot bp value
            SPTI = trapz(sys_phase);    % Systolic pressure�time index (tension�time index)
            DPTI = trapz(dias_phase);   % Diastolic pressure�time index
            SEVR = DPTI / SPTI ;        % Subendocardial viability ratio

            % Amplification Phenom.
            %Ampli = PSBP - CSBP; % Periferal BP - Central BP
            %Pulse pressure amplification
            %PPA = (PPP - CPP)/CPP;
            FF = (MPP /PP);
            
            features(1).left = left;
            features(1).right = right;
            features(1).SBP = SBP;
            features(1).DBP = DBP;
            features(1).ESBP = ESBP;
            features(1).skew = skew;
            features(1).PP = PP;
            features(1).Ti = Ti;
            features(1).Pi = Pi;
            features(1).AP = AP;
            features(1).AIx = AIx;
            features(1).MAP = MAP;
            features(1).MPP = MPP;
            features(1).MSBP = MSBP;
            features(1).MDBP = MDBP;
            features(1).iTTSP = iTTSP;
            features(1).TTSP = TTSP;
            features(1).iLVET = iLVET;
            features(1).LVET = LVET;
            features(1).iDT = iDT;
            features(1).DT = DT;
            features(1).HP = HP;
            features(1).DTF = DTF;
            features(1).SPTI = SPTI;
            features(1).DPTI = DPTI ; 
            features(1).SEVR = SEVR ;
            %features(1).Ampli = Ampli;
            %features(1).PPA = PPA;
            features(1).FF = FF;
            
            %features = struct2table(features,'AsArray',true);
            stats = [];
    end
    
    
end

function control_flag = control_SBP_DBP(SBP,DBP,dbp_lb,sbp_lb,dbp_ub,sbp_ub)
    %Extreme SBP values
    if SBP < sbp_lb || SBP > sbp_ub % Bad signal
        control_flag = 1;
        return;
    end
    %Extreme small diff SBP - DBP
    if SBP - DBP <10
       control_flag = 1;
        return;
    end
        
    %Extreme SBP values
    if DBP < dbp_lb || DBP > dbp_ub % Bad signal
        control_flag = 1;
        return;
    end
    control_flag = 0;
end

