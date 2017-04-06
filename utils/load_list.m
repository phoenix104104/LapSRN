function [list, kWs] = load_list(list_name)

    f = fopen(list_name);
    if( f == -1 )
        error('%s does not exist!', list_name);
    end
    C = textscan(f, '%s %d', 'CommentStyle', '#');
    list = C{1};
    kWs  = double(C{2});
    fclose(f);
end