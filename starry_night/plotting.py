import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import cm, gridspec, ticker
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import numpy as np


def exponential(x, m, b):
    return np.exp(m * x + b)


def plot_kernel_curve(stars, config, args):
    res = list()
    gr = stars.query('kernel>=1 & vmag<4').reset_index().groupby('HIP')
    ax = plt.figure().add_subplot(111)
    color = cm.jet(np.linspace(0,1,10 * (gr.vmag.max().max()-gr.vmag.min().min())+2 ))
    for _, s in gr:
        # normalize
        n = s.response_orig.max()
        res.append(s.query('response_orig == {}'.format(n)).kernel.values)
        plt.plot(s.kernel.values, s.response_orig.values/n, marker='o', c=color[int(round(s.vmag.max()*10))])
    ax.set_xlim(0., 5.1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('$\sigma$ of LoG filter')
    ax.set_ylabel('Kernel response normalized')
    lEntry = Line2D([], [], color='black', marker='o', markersize=6, label='Response of all stars')
    ax.grid()
    ax.legend(handles=[lEntry])
    if args['-s']:
        plt.savefig('kernel_curve_{}.png'.format(config['properties']['name']))
    if args['-v']:
        plt.show()
    plt.close('all')


def plot_choose_sigma(stars, kernelSize, config, args):
    gr = stars.query('kernel>=1 & vmag<4').reset_index().groupby('kernel')
    res = list()
    fig = plt.figure(figsize=(16,10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax = fig.add_subplot(gs[0])
    ax.grid()
    for _, s in gr:
        # dont plot faint stars
        popt, pcov = curve_fit(exponential, s.vmag.values, s.response_orig.values)
        res.append((s.kernel.max(),*popt, np.sqrt(pcov[0,0]), np.sqrt(pcov[1,1])))
    res = np.array(res)
    ax.scatter(res[:,0],res[:,2], label='b')
    ax.set_xlabel('$\sigma$ of LoG filter')
    ax.set_ylabel('b')
    ax.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(b=True, which='major')
    ax.grid(b=True, which='minor')
    ax2 = plt.subplot(gs[1], sharex=ax)
    ax2.scatter(res[:,0],res[:,4], label='b')
    ax2.set_ylabel('Standard deviation')
    ax2.set_xlim((kernelSize[0]*0.9, kernelSize[-1]+.1))
    ax2.set_ylim((np.min(res[:,4])*0.95, np.max(res[:,4])*1.08))
    ax2.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.grid(b=True, which='major')
    ax2.grid(b=True, which='minor')
    plt.tight_layout()
    if args['-s']:
        plt.savefig('chose_sigma_{}.png'.format(config['properties']['name']))
    if args['-v']:
        plt.show()
    plt.close('all')


def plot_camera_image(img, images, output, stars, celObjects, config, args):
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    vmin = np.nanpercentile(img, 5)
    vmax = np.nanpercentile(img, 90.)
    ax.imshow(img, vmin=vmin,vmax=vmax, cmap='gray')
    cax = ax.scatter(stars.x.values, stars.y.values, c=stars.visible.values, cmap = plt.cm.RdYlGn, s=30, vmin=0, vmax=1)
    celObjects['points_of_interest'].plot.scatter(x='x', y='y', ax=plt.gca(), s=80, color='white', marker='^', label='Sources')
    ax.text(
        0.98, 0.02, str(output['timestamp']),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        backgroundcolor='black',
        color='white', fontsize=15,
    )
    cbar = fig.colorbar(cax)
    cbar.ax.set_ylabel('Visibility')

    if args['-s']:
        plt.savefig('cam_image_{}.png'.format(images['timestamp'].isoformat()))
    if args['--daemon']:
        plt.savefig('cam_image_{}.png'.format(config['properties']['name']), dpi=300)
    if args['-v']:
        plt.show()
    plt.close('all')


def plot_kernel_response(llim, ulim, img, images, data, stars, config, args):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.semilogy()

    # draw visibility limits
    x = np.linspace(-5+stars.vmag.min(), stars.vmag.max()+5, 20)
    y1 = 10**(x*llim[0] + llim[1])
    y2 = 10**(x*ulim[0] + ulim[1])
    ax.plot(x, y1, c='red', label='lower limit')
    ax.plot(x, y2, c='green', label='upper limit')

    stars.plot.scatter(x='vmag', y='response', ax=ax, logy=True, c=stars.visible.values,
            cmap = plt.cm.RdYlGn, grid=True, vmin=0, vmax=1, label='Kernel Response', s=40)
    ax.set_xlim((-1, float(data['vmaglimit'])+0.5))
    ax.set_ylim((
            10**(llim[0]*float(config['analysis']['vmaglimit'])+llim[1]-1),
            10**(ulim[0]*-1+ulim[1])
            ))
    ax.set_ylabel('Kernel Response')
    ax.set_xlabel('Star Magnitude')
    if args['-c'] == 'GTC':
        if args['--function'] == 'Grad':
            ax.axhspan(ymin=11**2/255**2, ymax=13**2/255**2, color='red', alpha=0.5, label='old threshold range')
        ax.axvline(4.5, color='black', label='Magnitude lower limit')

    # show camera image in a subplot
    ax_in= inset_axes(ax,
            width='30%',
            height='40%',
            loc=3)
    vmin = np.nanpercentile(img, 0.5)
    vmax = np.nanpercentile(img, 99.)
    ax_in.imshow(img,cmap='gray',vmin=vmin,vmax=vmax)
    color = cm.RdYlGn(stars.visible.values)
    stars.plot.scatter(x='x',y='y', ax=ax_in, c=color, vmin=0, vmax=1, grid=True)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)

    leg = ax.legend(loc='best')
    leg.legendHandles[2].set_color('yellow')
    plt.tight_layout()
    if args['-s']:
        plt.savefig('response_{}_{}.png'.format(args['--function'], images['timestamp'].isoformat()))
    if args['--daemon']:
        plt.savefig('response_{}.png'.format(config['properties']['name']),dpi=200)
    if args['-v']:
        plt.show()
    plt.close('all')


def plot_ratescan(response, sobelList, logList, gradList, minThresholdPos, args):
    fig = plt.figure(figsize=(19.2,10.8))
    ax1 = fig.add_subplot(111)
    plt.xscale('log')
    plt.grid()
    ax1.plot(response, sobelList[:, 0], marker='x', c='blue', label='Sobel Kernel - Percent')
    ax1.plot(response, logList[:, 0], marker='x', c='red', label='log Kernel - Percent')
    ax1.plot(response, gradList[:, 0], marker='x', c='green', label='Square Gradient - Percent')
    ax1.axvline(response[minThresholdPos[0]], color='green')
    ax1.axvline(response[minThresholdPos[1]], color='blue')
    ax1.axvline(response[minThresholdPos[2]], color='red')
    ax1.axvline(14**2 / 255**2, color='black', label='old threshold')
    ax1.set_ylabel('')
    ax1.legend(loc='center left')

    ax2 = ax1.twinx()
    # ax2.plot(response, gradList[:,1], marker='o', c='green', label='Square Gradient - Pixcount')
    # ax2.plot(response, sobelList[:,1], marker='o', c='blue', label='Sobel Kernel - Pixcount')
    # ax2.plot(response, logList[:,1], marker='o', c='red', label='log Kernel - Pixcount')
    ax2.plot(response, gradList[:,2], marker='s', c='green', label='Square Gradient - Clustercount')
    ax2.plot(response, sobelList[:,2], marker='s', c='blue', label='Sobel Kernel - Clustercount')
    ax2.plot(response, logList[:,2], marker='s', c='red', label='log Kernel - Clustercount')
    ax2.axhline(gradList[minThresholdPos[0],2], color='green')
    ax2.axhline(sobelList[minThresholdPos[1],2], color='blue')
    ax2.axhline(logList[minThresholdPos[2],2], color='red')
    ax2.legend(loc='upper right')
    ax2.set_xlim((min(response), max(response)))
    ax2.set_ylim((0,16000))
    if args['-v']:
        plt.show()
    if args['-s']:
        plt.savefig('rateScan.pdf')
    plt.close('all')


def plot_cloudmap_and_image(img, images, cloud_map, output, args):
    fig = plt.figure(figsize=(16,9))
    ax1 = fig.add_subplot(121)
    vmin = np.nanpercentile(img, 5.5)
    vmax = np.nanpercentile(img, 99.9)
    ax1.imshow(img, vmin=vmin, vmax=vmax, cmap='gray', interpolation='none')
    ax1.set_ylabel('$y$ / px')
    ax1.grid()
    ax1.text(
        0.98, 0.02, str(output['timestamp']),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax1.transAxes,
        backgroundcolor='black',
        color='white', fontsize=15,
    )
    ax2 = fig.add_subplot(122)
    ax2.imshow(cloud_map, cmap='gray_r', vmin=0, vmax=1)
    ax2.grid()
    ax2.set_yticks([])
    fig.text(0.53, 0.02, '$x$ / px', ha='center')
    plt.tight_layout(h_pad=-0.1)
    if args['-s']:
        plt.savefig('cloudMap_{}.png'.format(images['timestamp'].isoformat()))
    if args['-v']:
        plt.show()
    plt.close('all')


def plot_cloudmap(cloud_map, config):
    ax = plt.subplot(111)
    ax.imshow(cloud_map, cmap='gray_r', vmin=0, vmax=1)
    ax.grid()
    plt.savefig('cloudMap_{}.png'.format(config['properties']['name']), dpi=400)
