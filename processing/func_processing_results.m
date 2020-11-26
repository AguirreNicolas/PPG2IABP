function [struct_summarized_process] = func_processing_results(n_files,channels_selected)
%{
Output:
    * id: subject ID
    * status: NaN = Not Proceesd, 0 = Fail , 1 = Ok
    * count_cycles: Qty cycles consumed in processing
    * count_f = Fails due to flatness in signal
    * count_p =  Fails due to fails in peaks/valleys (saturated)
    * count_fe = Fails due to feature extraction
%}
ppg = ~isempty(find(strcmp('PLETH',cellstr(channels_selected))));
abp = ~isempty(find(strcmp('ABP',cellstr(channels_selected))));
rbp = ~isempty(find(strcmp('RBP',cellstr(channels_selected))));
fbp = ~isempty(find(strcmp('FBP',cellstr(channels_selected))));
%ecg = ~isempty(find(strcmp('II',cellstr(channels_selected))));


for i= n_files:-1:1
    struct_summarized_process(i).id = NaN;
    struct_summarized_process(i).status = NaN;
    struct_summarized_process(i).count_cycles = NaN;
    struct_summarized_process(i).count_f = NaN;
    struct_summarized_process(i).count_p = NaN;
    struct_summarized_process(i).count_fe = NaN;
    
    %Fail causes
    if ppg
        struct_summarized_process(i).count_fe_ppg = NaN;
    end
    if abp
        struct_summarized_process(i).count_fe_abp = NaN;
    end
    %if ecg
    %    struct_summarized_process(i).count_fe_ecg = NaN;
    %end
    if rbp
        struct_summarized_process(i).count_fe_rbp = NaN;
    end
    if fbp
        struct_summarized_process(i).count_fe_fbp = NaN;
    end    
    
end

end

