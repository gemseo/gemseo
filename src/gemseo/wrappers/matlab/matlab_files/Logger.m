classdef Logger
    %LOGGER Summary of this class goes here
    %   Detailed explanation goes here

    properties (Access = private)
        info_logger = '    INFO - ';
        warning_logger = ' WARNING - ';
        error_logger = '   ERROR - ';
    end

    methods
        function this = Logger()
        end
    end
    methods
        function info(varargin)
            p =inputParser;
            addRequired(p, 'this');
            addRequired(p, 'message',@(x)ischar(x));
            parse(p, varargin{:});
            this = p.Results.this;
            time_str = this.get_time();
            disp([this.info_logger time_str ' : ' p.Results.message])
        end
        function warning(varargin)
            p =inputParser;
            addRequired(p, 'this');
            addRequired(p, 'message',@(x)ischar(x));
            parse(p, varargin{:});
            this = p.Results.this;
            time_str = this.get_time();
            fprintf(2,[this.warning_logger time_str ' : ' p.Results.message '\n'])
        end
        function error(varargin)
            p =inputParser;
            addRequired(p, 'this');
            addRequired(p, 'message',@(x)ischar(x));
            parse(p, varargin{:});
            this = p.Results.this;
            time_str = this.get_time();
            fprintf(2,[this.error_logger time_str ' : ' p.Results.message '\n'])
        end
    end
    methods (Access = private)
        function time_str = get_time(this)
            c = clock;
            hour = num2str(c(4));
            min = num2str(c(5));
            sec = num2str(floor(c(6)));
            time={hour,min,sec};
            for index_time= 1:3,
                if length(time{index_time})<2
                    time{index_time} = ['0' time{index_time}];
                end
            end
            time_str = [time{1} ':' time{2} ':' time{3}];
        end
    end
end
