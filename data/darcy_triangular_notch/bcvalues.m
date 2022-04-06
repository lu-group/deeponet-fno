function bcmatrix = bcvalues(location, state)
    
    global bc_cond
    
    N = 101;    
    xf = linspace(0, 1, N);
    bcmatrix = interp1(xf, bc_cond, location.x);
end