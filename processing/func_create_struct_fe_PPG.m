function [struct_feature_PPG] = func_create_struct_fe_PPG(q)
            
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % by Nicolas Aguirre    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Struct created to pre-alloacte the features from PPG


    for i = q:-1:1
        struct_feature_PPG(i).left_margin = NaN;
        struct_feature_PPG(i).right_margin = NaN;
        struct_feature_PPG(i).sys_peak = NaN;
        struct_feature_PPG(i).dia_start = NaN;
        struct_feature_PPG(i).dia_peak = NaN;
        struct_feature_PPG(i).left = NaN; % Index Start of pulse
        struct_feature_PPG(i).right = NaN; % Index End of pulse 
        struct_feature_PPG(i).sys_idx = NaN; % Index Systolic peak
        struct_feature_PPG(i).dicN_idx = NaN; % Index DNotch
        struct_feature_PPG(i).skew = NaN;%Skewness        
        struct_feature_PPG(i).w = NaN;
        struct_feature_PPG(i).y = NaN;
        struct_feature_PPG(i).z = NaN;
        struct_feature_PPG(i).a = NaN;
        struct_feature_PPG(i).b = NaN;
        struct_feature_PPG(i).c = NaN;
        struct_feature_PPG(i).d = NaN;
        struct_feature_PPG(i).e = NaN;
        %Ratios
        struct_feature_PPG(i).bd = NaN;
        struct_feature_PPG(i).scoo = NaN;
        struct_feature_PPG(i).sc = NaN;
        struct_feature_PPG(i).bcda = NaN;
        struct_feature_PPG(i).cw = NaN;
        struct_feature_PPG(i).bcdea = NaN;
        struct_feature_PPG(i).sdoo = NaN;
        struct_feature_PPG(i).cs = NaN;
        %Time
        struct_feature_PPG(i).tc = NaN;   % complete time
        struct_feature_PPG(i).ts = NaN;    % time to sys peak
        struct_feature_PPG(i).td = NaN;   % time from sys peak till the end
        struct_feature_PPG(i).tod = NaN;  % time to dia-peak
        struct_feature_PPG(i).tnt = NaN;  % time from sys peak to dia peak
        struct_feature_PPG(i).ttn = NaN;  % time from dia peak till the end
        %Areas
        struct_feature_PPG(i).auc_sys = NaN;
        struct_feature_PPG(i).aac_sys = NaN;
        struct_feature_PPG(i).auc_dia = NaN;
        struct_feature_PPG(i).aac_dia = NaN;
        %Target
        struct_feature_PPG(i).mean_sbp = NaN;
        struct_feature_PPG(i).mean_dbp = NaN;
        struct_feature_PPG(i).sbp_class = NaN; 
        struct_feature_PPG(i).dbp_class = NaN;
    end
end

