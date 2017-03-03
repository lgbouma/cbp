import matplotlib.pyplot as plt
import numpy as np

def add_subplot_axes(ax,fig,rect,axisbg='w'):
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def example2():
    fig, axes = plt.subplots(figsize=(10,10),nrows=2,ncols=2)
    axes = axes.flatten()
    subpos = [0.05,0.79,0.4,0.2]
    x = np.linspace(-np.pi,np.pi)
    for ax in axes:
        ax.set_xlim(-np.pi,np.pi)
        ax.set_ylim(-1,3)
        ax.plot(x,np.sin(x))
        subax1 = add_subplot_axes(ax,fig,subpos)
        subax2 = add_subplot_axes(subax1,fig,subpos)
        subax1.plot(x,np.sin(x))
        subax2.plot(x,np.sin(x))

    plt.savefig('insetax.png', dpi=300)

if __name__ == '__main__':
    example2()
