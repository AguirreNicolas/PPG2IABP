function [] = func_plot_delineator(data,foot,SysPeak,DicroN,column_name)

    if istable(data)
        col_names = data.Properties.VariableNames;
        q_channels_bp = size(col_names,2);
        f = figure();
        bFig(1:q_channels_bp) = axes(f);
        q=1;
        for i_bp = col_names
            bFig(q) = subplot(q_channels_bp,1,q);
            hold on
            plot(data.(i_bp{1}),'k');
            for i = foot.(i_bp{1})
                scatter(i,data.(i_bp{1})(i),'og','filled');
                %line([i i],[min(data.(i_bp{1})) max(data.(i_bp{1}))],'Color',[0 0 0])
            end
            for i = SysPeak.(i_bp{1})
                scatter(i,data.(i_bp{1})(i),'or','filled')
                %line([i i],[min(data.(i_bp{1})) max(data.(i_bp{1}))],'Color',[0 1 0])
            end
            for i = DicroN.(i_bp{1})
                scatter(i,data.(i_bp{1})(i),'ob','filled')
                %line([i i],[min(data.(i_bp{1})) max(data.(i_bp{1}))],'Color',[0 0 1])
            end
            hold off
            title((i_bp{1}));
            q=q+1;
        end
        for i=1:q_channels_bp-1
            linkaxes([bFig(1), bFig(i+1)], 'x');
        end
    else
        q_channels_bp = min(size(data));
        figure;
        q=1;
        col_names = fieldnames(foot);
        if nargin == 5
           col_names =column_name;  
        end
        for i_bp = 1:q_channels_bp
            subplot(q_channels_bp,1,q);
            hold on
            plot(data(i_bp,:),'k');
            for i = getfield(foot,char(col_names(i_bp))) %#ok<*GFLD>
                %getfield(foot,string(col_names(i_bp))
                scatter(i,data(i_bp,i),'og','filled');
                %line([i i],[min(data(i_bp,:)) max(data(i_bp,:))],'Color',[1 0 0])
            end
            for i =  getfield(SysPeak,char(col_names(i_bp)))
                %getfield(SysPeak,string(col_names(i_bp)))
                scatter(i,data(i_bp,i),'or','filled');
                %line([i i],[min(data(i_bp,:)) max(data(i_bp,:))],'Color',[0 1 0])
            end
            for i =  getfield(DicroN,char(col_names(i_bp)))
                %getfield(DicroN,string(col_names(i_bp)))
                scatter(i,data(i_bp,i),'ob','filled');
                %line([i i],[min(data(i_bp,:)) max(data(i_bp,:))],'Color',[0 0 1])
            end
            %legend(h_foot,h_sys,h_dn,'onstep','Sys_Peak','DicroN');
            hold off
            title(i_bp);
            q=q+1;
        end

    end

end