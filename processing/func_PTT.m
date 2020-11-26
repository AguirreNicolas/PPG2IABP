function[stats_PTT, feature_PTT, raw_signals] = func_PTT(method,ecg,bp1,bp2,qrs_i_raw,point1,point2,fs,show)
%{
Input:
    method: ECG-zero time or BP's delta time
    ecg: ECG signal
    bp1: ABP I signal
    bp2: ABP II signal
    qrs_i_raw: index of QRS
    point1: foots/systolic of ABP I pulses
    point2: foots/systolic of ABP II pulses

Output:
    stats_PTT: struct with mean_PTT and std_PTT
    feature_PTT: struct with indexes, foots and PTT
    raw_signals: contain the raw signals (beat-by-beat)
%}

if method == 1
    q_struct = size(qrs_i_raw,2);
else
    q_struct = size(point1,2);
end

control_vector = true(1,q_struct); 
feature_PTT = func_create_struct_PTT(q_struct);

%Thresholds
PTT_min = 30; % in ms
PTT_max = 160; % in ms
len_limit = 2*fs;
p_ok_pulses = 0.5;
sd_lim = 2;

if method == 1 %ECG Method
    raw_signals = zeros(q_struct,3,2*fs+1);
    for i_R = 1:(q_struct-1)
        %Index beats
        left = qrs_i_raw(i_R);
        right = qrs_i_raw(i_R+1);
        if left + len_limit < right
            continue
        end
        abp_point1 = point1((point1>left) & (point1<right));
        if isempty(abp_point1)
            continue
        end
        abp_point1 = abp_point1(1); %If there is more than 1 foot (bad R-R peaks)
        abp_point2 = point2((point2>left) & (point2<right) & (abp_point1<point2));
        if isempty(abp_point2)
            continue
        end

        %Sections
        ecg_section = ecg(left:right);
        bp1_section = bp1(left:right);
        bp2_section = bp2(left:right);
        
        %PTT
        diff1 = ((abp_point1 - qrs_i_raw(i_R))/fs)*1000; %--> in ms
        diff2 = ((abp_point2 - qrs_i_raw(i_R))/fs)*1000;%--> in ms
        PTT_new = diff2(1)-diff1(1);

        if PTT_new < PTT_min %|| PTT_new > PTT_max %Extremes value
            continue
        end
        %Save to struct
        feature_PTT(i_R).PTT = PTT_new;
        feature_PTT(i_R).left = left;
        feature_PTT(i_R).abp_point1 = abp_point1;
        feature_PTT(i_R).abp_point2 = abp_point2;
        feature_PTT(i_R).diff1 = diff1;
        feature_PTT(i_R).diff2 = diff2;
        feature_PTT(i_R).right = right;

        %Beat-by-Beat
        raw_new = zeros(1,3,2*fs+1);
        %Centered
        %padding = ceil((size(raw_new,3) - size(ecg_section,2))/2); %Q zeros a cada costado de la señal centrada
        %raw_new(1,:,padding+1:padding+size(ecg_section,2)) = [ecg_section;bp1_section;bp2_section];% 3 = ECG-BP1-BP2
        %No-Centered
        raw_new(1,:,1:size(ecg_section,2)) = [ecg_section;bp1_section;bp2_section];

        raw_signals(i_R,:,:) = raw_new;

    end      

else %ABP Method
    raw_signals = zeros(q_struct,2,2*fs+1);
    for i_foot1 = 1:(q_struct-1)
        %Index beats
        left = point1(i_foot1);
        right = point1(i_foot1+1);
        if left + len_limit < right
            continue
        end
        abp_point2 = point2((point2>left) & (point2<right) & (abp_point1<point2));
        if isempty(abp_point2)
            continue
        end
        
        %Sections
        bp1_section = bp1(left:right);
        bp2_section = bp2(left:right);
        
        %PTT
        PTT_new = ((abp_point2(1) - left)/fs)*1000;
        if PTT_new < PTT_min || PTT_new > PTT_max  %Extreme value
            continue
        end
        
        %Save to struct
        feature_PTT(i_foot1).PTT = PTT_new;
        feature_PTT(i_foot1).left = left;
        feature_PTT(i_foot1).abp_point1 = left;
        feature_PTT(i_foot1).abp_point2 = abp_point2;
        feature_PTT(i_foot1).right = right;

        %Beat-by-Beat            
        raw_new = zeros(1,2,2*fs+1);
        %Centered
        %padding = ceil((size(raw_new,3) - size(bp1_section,2))/2); %Q zeros a cada costado de la señal centrada
        %raw_new(1,:,padding+1:padding+size(bp1_section,2)) = [bp1_section;bp2_section];% 3 = ECG-BP1-BP2
        %No-Centered
        raw_new(1,:,1:size(bp1_section,2)) = [bp1_section;bp2_section];
        raw_signals(i_foot1,:,:) = raw_new;
    end
end

%Eliminate NaN Pulses
for i_Cycle = 1:q_struct
    for fn = fieldnames(feature_PTT)'
        fieldcontent = feature_PTT(i_Cycle).(fn{1});
        if isnan(fieldcontent)
            control_vector(1,i_Cycle)=false;
        end
    end
end
feature_PTT = feature_PTT(control_vector);
raw_signals = raw_signals(control_vector,:,:);
%Control
if size((feature_PTT),2) < q_struct*p_ok_pulses
    feature_PTT = [];
    raw_signals=[];
    stats_PTT=[];
    return;
end
%Stats Struct
values = [feature_PTT.PTT];
std_v = std(values);
mean_PTT = mean(values);
lb = mean_PTT -sd_lim*std_v;
ub = mean_PTT + sd_lim*std_v;
mean_PTT = mean(values(values<ub & values>lb));
std_PTT = std(values(values<ub & values>lb));
stats_PTT.mean_PTT = mean_PTT;
stats_PTT.std_PTT = std_PTT;

if(show)
    plot_PTT(method,feature_PTT,raw_signals)
end
    
end
    




function [struct_feature_PTT] = func_create_struct_PTT(q_struct)

for i= q_struct:-1:1
    struct_feature_PTT(i).left = NaN;
    struct_feature_PTT(i).right = NaN;
    struct_feature_PTT(i).abp_point1 = NaN;
    struct_feature_PTT(i).abp_point2 = NaN;
    struct_feature_PTT(i).diff1 = NaN;   
    struct_feature_PTT(i).diff2 = NaN;
    struct_feature_PTT(i).PTT =NaN; 
end

end

function [] = plot_PTT(method,feature_PTT, raw_signals)

    values = [feature_PTT.PTT];
    if method==1
        for i = 1:size(values,2)
            i_ecg_left = feature_PTT(i).left;
            i_ecg_right = feature_PTT(i).right;
            len_ecg = i_ecg_right - i_ecg_left;
            i_bp1 = feature_PTT(i).abp_point1 - feature_PTT(i).left;
            i_bp2 = feature_PTT(i).abp_point2 - feature_PTT(i).left;
            s_ecg = normalize_signal(squeeze(raw_signals(i,1,:)));
            s_bp1 = normalize_signal(squeeze(raw_signals(i,2,:)));
            s_bp2 = normalize_signal(squeeze(raw_signals(i,3,:)));
            figure();
            hold on
            plot (s_ecg,'k');
            plot (s_bp1,'r');
            plot (s_bp2,'b');
            xlim([0 len_ecg])
            scatter(i_bp1,s_bp1(i_bp1),'or','filled');
            scatter(i_bp2,s_bp2(i_bp2),'ob','filled');
            legend('ECG','BP1','BP2','foot1','foot2');
            hold off
        end
        
    else
        
        for i = 1:size(values,2)
            i_left = feature_PTT(i).left;
            i_right = feature_PTT(i).right;
            len_abp = i_right - i_left;
            i_bp1 = 1;
            i_bp2 = feature_PTT(i).abp_point2 - feature_PTT(i).left;
            s_bp1 = normalize_signal(squeeze(raw_signals(i,1,:)));
            s_bp2 = normalize_signal(squeeze(raw_signals(i,2,:)));
            figure();
            hold on
            plot (s_bp1,'r');
            plot (s_bp2,'b');
            xlim([0 len_abp])
            scatter(i_bp1,s_bp1(i_bp1),'or','filled');
            scatter(i_bp2,s_bp2(i_bp2),'ob','filled');
            legend('BP1','BP2','foot1','foot2');
            hold off
        end
        
    end
    
end

