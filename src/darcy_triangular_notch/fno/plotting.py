"""
Author: Somdatta Goswami, somdatta_goswami@brown.edu
Plotting xDisplacement, yDisplacement and phi
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.use('Agg')

def plotField(k_print, disp_pred, disp_true, istep, folder, segment):

        fig = plt.figure(constrained_layout=False, figsize=(10,3))
        fig.suptitle(segment+'ing case: ' + str(istep+1), fontsize=16)
        gs = fig.add_gridspec(1,4)
        plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.5, wspace = 0.4, hspace = 0.1)
        
        ax = fig.add_subplot(gs[0,0])        
        h = ax.imshow(k_print, origin='lower', interpolation='nearest', cmap='jet')
        ax.set_title('K(x)')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        divider = make_axes_locatable(ax)

        disp_max = np.max(np.maximum(disp_pred,disp_true))
        disp_min = np.min(np.minimum(disp_pred,disp_true)) 
        
        ax = fig.add_subplot(gs[0,1])        
        h = ax.imshow(disp_pred,origin='lower', interpolation='nearest', cmap='jet')
        ax.set_title('Pred h(x)')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
#        h.set_clim(vmin=0.0, vmax = 0.6)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        ax = fig.add_subplot(gs[0,2])
        h = ax.imshow(disp_true,origin='lower', interpolation='nearest', cmap='jet')
        ax.set_title('True h(x)')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
#        h.set_clim(vmin=0.0, vmax = 0.6)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)
        
        ax = fig.add_subplot(gs[0,3])
        h = ax.imshow(abs(disp_pred - disp_true),origin='lower', interpolation='nearest', cmap='jet')
        ax.set_title('Error')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)        
#        h.set_clim(vmin=0, vmax=0.006)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        fig.savefig(folder + '/step_' + str(istep) + '.png')
        plt.close()
    
