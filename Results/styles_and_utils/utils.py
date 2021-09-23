class ViT:
    def __init__(self, size, useline=True):
        if useline:
            self.marker = 'o-.'
        else:
            self.marker = 'o'
        self.color = '#1f77b4'
        
        if size == 's':
            self.ms = 4
            self.oc = 0.95
            self.hatch = 'x'
            self.label = 'ViT-8M/10'
        
        elif size == 'm':
            self.ms = 10
            self.oc = 0.75
            self.hatch = 'x'
            self.label = 'ViT-32M/10'            
            
        elif size == 'l':
            self.ms = 16
            self.oc = 0.55
            self.hatch = '..'
            self.label = 'ViT-60M/10'            

class Unet:
    def __init__(self, size, useline=True):
        if useline:
            self.marker = 's-.'
        else:
            self.marker = 's'
            
        self.color = '#bca1d1'
        
        if size == 's':
            self.ms = 4
            self.oc = 0.95
            self.hatch = 'x'
            self.label = 'U-Net-8M'
        
        elif size == 'm':
            self.ms = 10
            self.oc = 0.75
            self.hatch = 'x'
            self.label = 'U-Net-31M'            
            
        elif size == 'l':
            self.ms = 16
            self.oc = 0.55
            self.hatch = '..'
            self.label = 'U-Net-124M' 

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector 

def mark_inset(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

    pp = BboxPatch(rect, fill=False, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)

    return pp, p1, p2            

# ## Example
# # location for the zoomed portion 
# axins = ax.inset_axes([.54, .27, .4, .4]) 

# # plot the zoomed portion
# axins.plot(knee_train_data[1:], knee_vit_32Mp10[1:], 'o-.', linewidth=lw, markersize=ms1, alpha=oc1, color='#1f77b4', label='ViT-32M/10', zorder=4)
# axins.plot(knee_train_data[1:], knee_unet_124M[1:], 's-.', linewidth=lw, markersize=ms2, alpha=oc2, color='#bca1d1', label='U-Net-124M', zorder=3)
# axins.plot(knee_train_data[1:], knee_unet_31M[1:], 's-.', linewidth=lw, markersize=ms1, alpha=oc1, color='#bca1d1', label='U-Net-31M', zorder=2)
# axins.plot(knee_train_data[1:], knee_unet_8M[1:],'s-.', linewidth=lw, color='#bca1d1', label='U-Net-8M', zorder=1)
# # axins.grid('on')
# axins.set_xticks(knee_train_data[1:])
# axins.set_xticklabels(['17k', '35k'])
# axins.set_yticks([0.7425, 0.7435])
# axins.set_ylim([0.7420, 0.7442])

# mark_inset(ax, axins, loc1a=1, loc1b=4, loc2a=2, loc2b=3, fc="none", ec="#36d117",zorder=100)