import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import gridspec, ticker
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import numpy as np


def exponential(x, m, b):
    return np.exp(m * x + b)


def plot_kernel_curve(stars, outputfile):
    res = list()
    gr = stars.query('kernel>=1 & vmag<4').reset_index().groupby('HIP')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cm = plt.get_cmap()
    color = cm(np.linspace(0, 1, 10 * (gr.vmag.max().max() - gr.vmag.min().min()) + 2))
    for _, s in gr:
        # normalize
        n = s.response_orig.max()
        res.append(s.query('response_orig == {}'.format(n)).kernel.values)
        plt.plot(
            s.kernel.values,
            s.response_orig.values / n,
            c=color[int(round(s.vmag.max()*10))]
        )

    ax.set_xlabel('$\sigma$ of LoG filter')
    ax.set_ylabel('Kernel response normalized')
    lEntry = Line2D([], [], color='black', label='Response of all stars')
    ax.grid()
    ax.legend(handles=[lEntry])
    fig.tight_layout(pad=0)
    fig.savefig(outputfile, dpi=300)


def plot_choose_sigma(stars, kernelSize, outputfile):
    gr = stars.query('kernel>=1 & vmag<4').reset_index().groupby('kernel')

    res = list()
    for _, s in gr:
        popt, pcov = curve_fit(exponential, s.vmag.values, s.response_orig.values)
        res.append(
            (s.kernel.max(), *popt, np.sqrt(pcov[0, 0]), np.sqrt(pcov[1, 1]))
        )
    res = np.array(res)

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax1.grid()
    ax2.grid()

    ax1.scatter(res[:, 0], res[:, 2], label='b')
    ax1.set_xlabel('$\sigma$ of LoG filter')
    ax1.set_ylabel('b')

    ax2.scatter(res[:, 0], res[:, 4], label='b')
    ax2.set_ylabel('Standard deviation')

    fig.tight_layout(pad=0)
    fig.savefig(outputfile, dpi=300)


def plot_camera_image(img, timestamp, stars, celObjects, outputfile):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    vmin = np.nanpercentile(img, 5)
    vmax = np.nanpercentile(img, 90)

    for source in celObjects['points_of_interest'].itertuples():
        ax.plot(
            source.x,
            source.y,
            marker='^',
            color='C1',
        )
        ax.annotate(
            source.name,
            xy=(source.x, source.y),
            color='C1',
            xytext=(5, 0), textcoords='offset points',
            va='center',
        )

    ax.imshow(img, vmin=vmin,vmax=vmax, cmap='gray')
    ax.set_title(
        str(timestamp),
        verticalalignment='bottom', horizontalalignment='right',
    )

    plot = ax.scatter(
        stars.x.values,
        stars.y.values,
        c=stars.visible.values,
        cmap='RdYlGn',
        s=5,
        vmin=0,
        vmax=1,
    )
    fig.colorbar(plot, label='Visibility')
    fig.tight_layout(pad=0)
    fig.savefig(outputfile)


def plot_kernel_response(lower_limit, upper_limit, vmag_limit, img, data, stars, outputfile):
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
    ax.set_yscale('log')

    # draw visibility limits
    x = np.linspace(-5 + stars.vmag.min(), stars.vmag.max() + 5, 20)
    y1 = 10**(x * lower_limit[0] + lower_limit[1])
    y2 = 10**(x * upper_limit[0] + upper_limit[1])

    ax.plot(x, y1, c='red', label='lower limit')
    ax.plot(x, y2, c='green', label='upper limit')

    stars.plot.scatter(
        x='vmag',
        y='response',
        c=stars.visible.values,
        ax=ax,
        cmap=plt.cm.RdYlGn,
        vmin=0, vmax=1,
        label='Kernel Response',
    )
    ax.set_xlim((-1, float(data['vmaglimit']) + 0.5))
    ax.set_ylim(
        10**(lower_limit[0] * vmag_limit + lower_limit[1] - 1),
        10**(-upper_limit[0] + upper_limit[1])
    )

    ax.set_ylabel('Kernel Response')
    ax.set_xlabel('Star Magnitude')

    ax_in = inset_axes(ax, width='40%', height='40%', loc='lower left')

    vmin = np.nanpercentile(img, 0.5)
    vmax = np.nanpercentile(img, 99.5)
    ax_in.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)

    stars.plot.scatter(
        x='x',
        y='y',
        c='visible',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        s=3,
        colorbar=False,
        ax=ax_in,
    )
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)

    leg = ax.legend(loc='lower right')
    leg.legendHandles[2].set_color('yellow')

    fig.tight_layout(pad=0)
    fig.savefig(outputfile, dpi=300)

    plt.close('all')


def plot_ratescan(response, sobelList, logList, gradList, minThresholdPos, outputfile):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()

    ax1.set_xscale('log')
    ax1.grid()
    ax2.grid()

    kernels = [sobelList, logList, gradList]
    labels = ['Sobel Kernel', 'LoG Kernel', 'Square Gradient']

    for i, (kernel, label) in enumerate(zip(kernels, labels)):
        color = 'C{}'.format(i)
        ax1.plot(
            response,
            kernel[:, 0],
            color=color,
            marker='x',
            label='{} - Percent'.format(label),
        )
        ax2.plot(
            response,
            kernel[:, 2],
            color=color,
            marker='s',
            label='{} - Clustercount'.format(label)
        )

        ax1.axvline(response[minThresholdPos[i]], color=color)
        ax2.axhline(kernel[minThresholdPos[0], 2], color=color)

    ax1.axvline(14**2 / 255**2, color='black', label='old threshold')

    ax1.set_ylabel('')
    ax1.legend(loc='center left')
    ax2.legend(loc='upper right')

    ax2.set_ylim((0, 2**14))

    fig.tight_layout(pad=0)
    fig.savefig(outputfile)


def plot_cloudmap_and_image(img, cloud_map, timestamp, outputfile):
    fig = plt.figure()
    fig.suptitle(timestamp.isoformat())

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    vmin = np.nanpercentile(img, 5.5)
    vmax = np.nanpercentile(img, 99.9)

    ax1.imshow(img, vmin=vmin, vmax=vmax, cmap='gray', interpolation='none')
    ax2.imshow(cloud_map, cmap='gray_r', vmin=0, vmax=1)

    fig.tight_layout(pad=0)
    fig.savefig(outputfile, dpi=300)


def plot_cloudmap(cloud_map, outputfile):
    ax = plt.subplot(1, 1, 1)
    ax.imshow(cloud_map, cmap='gray_r', vmin=0, vmax=1)
    ax.grid()
    plt.savefig(outputfile, dpi=300)
