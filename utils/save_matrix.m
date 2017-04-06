function save_matrix(Q, filename, precision, print_message)
    
    if( ~exist('precision', 'var') )
        precision = 7;
    end
    if( ~exist('print_message', 'var') )
        print_message = 0;
    end
    file = fopen(filename, 'w');
    
    if( size(Q, 2) == 1 )
        dlmwrite(filename, Q, 'precision', precision, 'delimiter', '\n', 'newline', 'unix');
    else
        for i = 1:size(Q, 1)
            fprintf(file, sprintf('%%.%df\t', precision), Q(i, 1:end-1));
            fprintf(file, sprintf('%%.%df', precision), Q(i, end));
            fprintf(file, '\n');
        end
    end
    fclose(file);
    
    if( print_message )
        fprintf('Save %s\n', filename);
    end
end