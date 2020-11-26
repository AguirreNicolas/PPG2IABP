function[per_mat] = flat_lines(data,window, incline, show)

    % From:
    % https://github.com/gslapnicar/bp-estimation-mimic3/blob/master/cleaning_scripts/flat_lines.m
    %
    %Inputs:
    %   data ... 2xN matrix (containing signal PPG in the first and ABP in
    %   the second dimension)
    %   window ... size of the sliding window
    %   incline .. boolean, check for a small inclines yi == y(i+1) +- 1
    %   show ... boolean, show plots or not
    %outputs:
    %   per_ppg/abp ... percentage of points that are considered flat
    %
    %
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Modified by Nicolas Aguirre to perform with N channels.    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    % Flat line in ABP and PPG -> sliding window over the whole thing
    len = size(data,2);    
    q_channels = size(data,1);
    flat_locs_mat = ones(q_channels, (len-window +1));
    tmp_mat = zeros(q_channels, (len-window +1));
    
    %get the locations where i == i+1 == i+2 ... == i+window
    % efficient-ish sliding window
    for i = 2:(window)        
        for i_c = 1:q_channels
            tmp_mat(i_c,:) = (data(i_c,1:(len-window+1)) ==  data(i_c,i:(len-window+i))); 
        end
        
        %tmp_abp = (data(2,1:(len-window+1)) ==  data(2,i:(len-window+i)));
        %tmp_ppg = (data(1,1:(len-window+1)) ==  data(1,i:(len-window+i)));
        
        %isequal(tmp_abp,tmp_mat(2,:))
        %isequal(tmp_ppg,tmp_mat(1,:))
        
        
        %can be generalized -> for loop, if so deisred
        if(incline)
            % +1
            tmp_abp2 = (data(2,1:(len-window+1)) ==  (data(2,i:(len-window+i))) +1);
            tmp_ppg2 = (data(1,1:(len-window+1)) ==  (data(1,i:(len-window+i))) +1);
            % -1
            tmp_abp3 = (data(2,1:(len-window+1)) ==  (data(2,i:(len-window+i))) -1);
            tmp_ppg3 = (data(1,1:(len-window+1)) ==  (data(1,i:(len-window+i))) -1);
            % OR
            tmp_abp = (tmp_abp | tmp_abp2 | tmp_abp3);
            tmp_ppg = (tmp_ppg | tmp_ppg2 | tmp_ppg3);
        end
                
        flat_locs_mat = (flat_locs_mat & tmp_mat);
        
    end
    
    %extend to be the same size as data   
    flat_locs_mat = [flat_locs_mat, zeros(q_channels,window-1)];   
    flat_locs_mat2 = flat_locs_mat;
    
    %mark the ends of the window
    for i = 2:(window)
        for i_c=1:q_channels
            flat_locs_mat(i_c,i:end) =  flat_locs_mat(i_c,i:end) | flat_locs_mat2(i_c,1:(end-i+1)); 
        end
    end
    % percentages
    
    per_mat = zeros(1,q_channels);
    for i_c=1:q_channels
        per_mat(i_c) = sum(flat_locs_mat(i_c,:))/len;
    end
    
    if(show)
        % plot the flat line points
        x = [1:1:len];
        for i_c = 1:q_channels
            subplot(q_channels,1,i_c)
            hold on
            plot(x,data(i_c,:),'black')
            scatter(x(flat_locs_mat(i_c,:)==1), data(i_c,flat_locs_mat(i_c,:)==1),'red')
            hold off
        end

        %subplot(2,1,2)
        %hold on
        %plot(x,data(2,:),'black')
        %scatter(x(flat_locs_abp==1), data(2,flat_locs_abp==1),'red')
        %hold off
    end
end