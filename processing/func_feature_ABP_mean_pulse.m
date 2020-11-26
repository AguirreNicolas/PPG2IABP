function [features,mean_pulse] = func_feature_ABP_mean_pulse(raw_signals,mean_feature,fs,show)
% This function calculate features based only in the mean signal
% This fuction call to "func_feature_ABP" as main proces and plot the
% output image based on the mean of delineator function.
    
    std_lim = 1.25; % std limit to consider outliers points
    %mean, upper bound,lowebound
    [mean_pulse,ub,lb] = mean_pulse_non_zeros(raw_signals,std_lim,'slow');
    
    %Plot
    if (show)
        f = figure();
        for i_p=1:size(raw_signals,1)
            hold on
            plot((1:round(mean_feature.iHP))/fs,raw_signals(i_p,1:round(mean_feature.iHP)), 'Color', [0.5 0.5 0.5],'LineWidth',0.5);
        end
        %Mean
        [v_max,i_sp] = max(mean_pulse);
        p2 = plot((1:i_sp)/fs,mean_pulse(1:i_sp),'r','LineWidth',3); %osnet-syst
        p3 = plot((i_sp:round(mean_feature.iLVET))/fs,mean_pulse(i_sp:round(mean_feature.iLVET)),'g','LineWidth',3);% syst-dn
        p4 = plot((round(mean_feature.iLVET):round(mean_feature.iHP))/fs,mean_pulse(round(mean_feature.iLVET):round(mean_feature.iHP)),'b','LineWidth',3);% dn-end

        
        % Bounds
        p5 = plot((1:round(mean_feature.iHP))/fs,ub(1:round(mean_feature.iHP)),'m--','LineWidth',2);
        p6 = plot((1:round(mean_feature.iHP))/fs,lb(1:round(mean_feature.iHP)),'m--','LineWidth',2);
        % Foot = 1, iLVET = DicroN , iHP = Heart period,
        %line([mean_feature.iTTSP mean_feature.iTTSP],[min(mean_pulse) max(mean_pulse)],'Color',[0 1 0])
        %line([mean_feature.iLVET mean_feature.iLVET],[min(mean_pulse) max(mean_pulse)],'Color',[0 0 1])
        %line([mean_feature.iHP mean_feature.iHP],[min(mean_pulse) max(mean_pulse)],'Color',[1 0 1])
        %ylim([50 160]);
        hold off
        ylim([mean_pulse(1)-5 v_max+5])
        xlim([1/fs (mean_feature.iHP-5)/fs]) %for plot purpose only

        title('Mean Pulse','FontSize', 20)
        ylabel('Blood Pressure (mmHg)', 'FontSize', 20)
        xlabel('Time (s)', 'FontSize', 20)
        legend([p2,p3,p4,p5],{'Onset - Systolic peak','Systolic peak - Dicrotic notch','Dicrotic notch - End','Bounds'},'FontSize',8)
    end
    
    %Feature --> method 'mean_pulse'
    [~, features,~] = func_features_ABP('mean_pulse',mean_pulse,mean_feature.iHP,mean_feature.iTTSP,mean_feature.iLVET,fs);
    
end

function [m,ub,lb] = mean_pulse_non_zeros(raw_signals,std_lim,method)
 
    switch method
        case 'fast'    % Mean Pulse (fast)
            mean_pulse = mean(raw_signals,1);
            sd_pulse = std(raw_signals,1);
            lb = mean_pulse - std_lim*sd_pulse;
            ub = mean_pulse + std_lim*sd_pulse;
            b = (raw_signals <= ub & raw_signals >= lb & raw_signals>0);% logical matrix
            m = sum(raw_signals.* b,1)./...
                sum(b, 1);
            m(isnan(m))=0;
    
        case 'slow' % Mean Pulse (hard)
            b = (raw_signals>0);% logical matrix
            m = sum((raw_signals.* b),1)./...
                sum(b, 1);
            m(isnan(m))=0;
        
            sd_pulse = sqrt(sum(((raw_signals.* b)-m).^2,1)./...
                sum(b, 1));
            
            lb = m - std_lim*sd_pulse;
            ub = m + std_lim*sd_pulse;

            % mean pulse given a sd (logical operator)
            b = (raw_signals >= 40 & raw_signals <= ub & raw_signals >= lb);% logical matrix
            m = sum(raw_signals.* b,1)./...
                sum(b, 1);
            
            m(isnan(m))=0;
                      
    end

end
