function licenseControl(varargin)
p = inputParser;
addRequired(p, 'license_list',@(x)iscell(x));
addOptional(p, 'pause_frac', 60,@(x)isnumeric(x));
addOptional(p, 'pause_const', 20,@(x)isnumeric(x));
parse(p, varargin{:});

pause_frac = p.Results.pause_frac;
pause_const = p.Results.pause_const;
i=0;
total_time = 0;
log = Logger();
toolbox_to_test = p.Results.license_list;
gathered = false(1, length(toolbox_to_test));

clc;

while true,
    test = true;
    index_to_remove = [];
    for index = 1:length(toolbox_to_test),
        status = license('test',toolbox_to_test{index});
        if ~status,
            msg = '" toolbox does not exist, proceeding without';
            log.error(['"' toolbox_to_test{index} msg]);
            index_to_remove = cat(2,index,index_to_remove);
        else
            is_valid = license('checkout',toolbox_to_test{index});
            test = test && is_valid;
            if is_valid && not(gathered(index)),
                text = [toolbox_to_test{index} ' was fetched'];
                log.info(text)
            end
        end
    end
    for index = 1:length(index_to_remove)
        data_index = index_to_remove(index);
        toolbox_to_test(data_index) = [];
        gathered(data_index) = [];
    end
    if test,
        break
    else
        i=i+1;
        log.warning(['try '  num2str(i) ' to get licenses'])
        fraction = randi(1000)/1000;
        time_str = pause_const+pause_frac*fraction;
        total_time=total_time+time_str;
        log.info(['pausing for ' num2str(time_str) ' seconds'])
        pause(time_str)
        if total_time<60,
            log.info(['Elapsed time: ' num2str(total_time) ' seconds'])
        elseif total_time<60*60
            log.info(['Elapsed time: ' num2str(total_time/60) ' mins'])
        else
            log.info(['Elapsed time: ' num2str(total_time/(60*60)) ' hours'])
        end
        log.info(' ')
    end

end
log.info('Fetching of all Matlab licenses completed')

end
