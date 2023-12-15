import matplotlib.pyplot as plt
from  matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np


def plot_obj(data, xy_val=None, obj=None, bxsize=60,
             xlim=None, ylim=None,
             vmin=300, vmax=15000, cmap=plt.cm.viridis):
    if xy_val is not None:
        x_val = xy_val[0]; y_val = xy_val[1]

        rownum=round(y_val); colnum = round(x_val)
        xy=(colnum-bxsize//2,rownum-bxsize//2); width=bxsize; height =bxsize
    elif obj is not None:
        loc = obj
        xy = (loc[1].start, loc[0].start)
        width = loc[1].stop - loc[1].start
        height = loc[0].stop - loc[0].start
        rownum = xy[1]+height//2
        colnum = xy[0]+width//2
    else:
        raise ValueError('one of xy_val or obj must be specified')
    
    fig = plt.figure(figsize=(12,12))

    rect = Rectangle(xy, width, height, color='red', lw=3)

    ax = fig.add_subplot(2,2,1)
    ax.imshow(data, origin='lower', aspect='auto', cmap=cmap,vmin=vmin, vmax=vmax)
    ax.add_collection(PatchCollection([rect], facecolor='none', lw=3,edgecolor='red'))
    ax.axhline(rownum, color='red', lw=1)
    ax.axvline(colnum, color='red', lw=1)
    ax.grid()

    ax = fig.add_subplot(2,2,3)
    ax.plot(data[rownum])
    ax.set_xlim(xlim)
    ax.axvline(colnum, color='red')
    ax.grid()
    ax.set_ylim(ylim)

    ax = fig.add_subplot(2,2,2)
    ax.plot(data[:, colnum], np.arange(data.shape[1]))
    # ax.set_ylim(0, hdu.data.shape[0])
    ax.axhline(rownum, color='red')
    ax.grid()
    ax.set_xlim(xlim)

    # create slice from rectangle
    slc = (slice(xy[1],xy[1]+height), slice(xy[0], xy[0]+width))
    y=np.arange(slc[0].start, slc[0].stop); x=np.arange(slc[1].start, slc[1].stop)
    XX,YY = np.meshgrid(x,y)

    ax = fig.add_subplot(2,2,4,projection="3d")
    ax.plot_surface(XX,YY, data[slc], antialiased=False, cmap=cmap, vmin=vmin, vmax=vmax)