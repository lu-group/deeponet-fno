clear;
clc;
close all

curDir = pwd;
folder_name = [curDir '/Predictions'];
mkdir(folder_name)

flag_plot = true;

% -------------------------------------------
% this is the unstructured mesh for the original data
% -------------------------------------------
model = createpde;
notch_left = 0.49;
notch_right = 0.51;
R1 = [3, 4, 0, 1, 0.5+1e-12, 0.5-1e-12, 0, 0, sqrt(3)/2, sqrt(3)/2]';
R2 = [3, 4, notch_left, notch_right, notch_right, notch_left, 0, 0, 0.4, 0.4]';
gm = [R1,R2];
sf = 'R1-R2';
ns = char('R1','R2');
ns = ns';
g = decsg(gm,sf,ns);
geometryFromEdges(model,g);
hmax = 0.03;
generateMesh(model,'Hmax',hmax);

% -------------------------------------------
% 100 testing samples
N_data = 20;
resultFileName = ['Results/darcy_triangular_DeepONet.mat'];
Result = load(resultFileName);
% to store the l2 errors
L2_mesh = zeros(N_data,1);

if flag_plot
    figure('color',[1,1,1],'Units',...
        'centimeters','Position',[10,5,30,15]);
end

% -------------------------------------------
% loop for all the testing examples
for idx = 1:N_data
    
    % -------------------------------------------
    % load the original data - unstructured mesh
    u_truth = reshape(Result.y_test(idx,:,:),[],1);    % Nx1
    
    % load the FNO result - unstructured mesh
    u_pred = double(squeeze(Result.y_pred(idx,:,:) ) )';
%     u_pred = u_truth-u_pred;
    % interpolation
    % (x_grid, y_grid) is the location of the FNO output
    % (xx, yy) is the location of the unstructured mesh
%     u_pred = griddata(x_grid(:),y_grid(:),u_pred_box(:),Data.xx,Data.yy,'natural');
%     u_pred = reshape(u_pred,[],1);      % Nx1
    
    L2_mesh(idx) = norm(u_truth-u_pred) / norm(u_truth);
    
    % -------------------------------------------
    % display (on the unstructured mesh)
    if flag_plot
        
        clim = [0, 2];
        subplot(131);
        pdeplot(model,'XYData',u_truth);
        colormap(jet); colorbar(); %caxis(clim);
        axis equal;
        xlim([0,1]); ylim([0,1])
        title('Truth','interpreter', 'latex', 'fontsize', 14);
        xlabel('$x$', 'interpreter', 'latex', 'fontsize', 14);
        ylabel('$y$', 'interpreter', 'latex', 'fontsize', 14);
        caxis(clim);
        colorbar('southoutside')
        set(gca, 'YTick', [0,0.5,1], 'YTickLabel', [0,0.5,1])
        box on;
        
        subplot(132);
        pdeplot(model,'XYData',u_pred);
        colormap(jet); colorbar(); %caxis(clim);
        axis equal;
        xlim([0,1]); ylim([0,1])
        title('DeepONet','interpreter', 'latex', 'fontsize', 14);
        xlabel('$x$', 'interpreter', 'latex', 'fontsize', 14);
        ylabel('$y$', 'interpreter', 'latex', 'fontsize', 14);
        caxis(clim);
        colorbar('southoutside')
        set(gca, 'YTick', [0,0.5,1], 'YTickLabel', [0,0.5,1])
        box on;
        
        subplot(133);
        clim = [0, 0.1];
        pdeplot(model,'XYData',abs(u_truth-u_pred) );
        colormap(jet); colorbar(); caxis(clim);
        axis equal;
        xlim([0,1]); ylim([0,1])
        title('Error','interpreter', 'latex', 'fontsize', 14);
        xlabel('$x$', 'interpreter', 'latex', 'fontsize', 14);
        ylabel('$y$', 'interpreter', 'latex', 'fontsize', 14);
        colorbar('southoutside')
        box on;
        
        saveas(gcf, [folder_name '/TestCase', num2str(idx),'.png'])
        pause(0.1);
    end
end

% -------------------------------------------
% mean l2 error of unstructured mesh
fprintf('L2 norm error: %.3e\n', mean(L2_mesh));



