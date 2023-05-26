import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import matplotlib
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def convert_background_to_transparent(image):
    from PIL import Image
    img = Image.fromarray(image.astype(np.uint8)[:, :, [2, 1, 0]])
    img = img.convert("RGBA")
    width = img.size[0]
    height = img.size[1]
    for x in range(0, width):  # process all pixels
        for y in range(0, height):
            data = img.getpixel((x, y))
            if data[0] == 0 and data[1] == 0 and data[2] == 0:
                img.putpixel((x, y), (255, 255, 255, 0))
    # img = np.array(img)
    return img


def scalar_to_color(val, min=None, max=None):
    if min is None:
        min = val.min()
    if max is None:
        max = val.max()

    norm = matplotlib.colors.Normalize(vmin=min, vmax=max, clip=True)
    # use jet colormap
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)

    return mapper.to_rgba(val)[:, :3]


def plot_point_cloud(point_clouds, colors, show=False, save_path=None):
    """
    # export DISPLAY=:0.0
    Args:
        point_clouds: list array of [N, 3]
        colors: list of str
        show: show the image if True
        save_path: string path of saving image
    """
    fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111, projection='3d')
    ax = plt.gca(projection='3d')
    for points, color in zip(point_clouds, colors):
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.1, c=color, cmap='rainbow')
    ax.view_init(azim=-90, elev=90)

    plt.margins(0, 0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
    if show:
        plt.show()
    plt.close()


def fig2data(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    image = cv2.resize(image, (image.shape[1] // 3, image.shape[0] // 3))

    return image[:, :, :3]


def display_joints(joints, img, save_path=None, show=False, dpi=300, type='smpl'):
    joints = (joints + 1.) * img.shape[0] * 0.5
    H, W = img.shape[:2]
    if type == 'smpl':
        pairs = [[0, 1], [1, 4], [4, 7], [7, 10], [0, 2], [2, 5], [5, 8], [8, 11], [0, 3], [3, 6], [6, 9], [9, 12],
                 [12, 15], [9, 14], [14, 17], [17, 19], [19, 21], [21, 23], [9, 13], [13, 16], [16, 18], [18, 20],
                 [20, 22]]
        colors_skeleton = ['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'g', 'g', 'g', 'g', 'g',
                           'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b']
    elif type == 'coco':
        pairs = [[8, 9], [10, 11], [11, 12], [2, 1], [1, 0], [13, 14], [14, 15],
                 [3, 4], [4, 5], [8, 7], [7, 6], [6, 2], [6, 3], [8, 12], [8, 13]]
        colors_skeleton = ['m', 'b', 'b', 'r', 'r', 'b', 'b', 'r', 'r', 'm', 'm', 'r', 'r', 'b', 'b']
    elif type == 'mpii':
        pairs = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [11, 12], [11, 13], [13, 15],
                 [12, 14], [14, 16], [6, 12], [5, 11]]
        colors_skeleton = ['y', 'y', 'y', 'y', 'b', 'b', 'b', 'b', 'b', 'r', 'r', 'r', 'r', 'r', 'm', 'm']
    elif type == 'labeled':
        pairs = [[0, 1], [1, 3], [3, 5], [5, 7], [0, 2], [2, 4], [4, 6], [6, 8], [0, 15], [15, 16],
                 [15, 9], [9, 11], [11, 13], [15, 10], [10, 12], [12, 14]]
        colors_skeleton = ['r'] * len(pairs)
    else:
        raise ValueError('pose ')
    fig = plt.figure(figsize=(W / 200, H / 200), dpi=dpi)
    plt.imshow(img)
    for i, idx in enumerate(pairs):
        plt.plot(joints[idx][:, 0], H - joints[idx][:, 1], 'r-', color=colors_skeleton[i], linewidth=1)
    plt.scatter(joints[:, 0], H - joints[:, 1], c=range(0, len(joints) * 10, 10), s=6)

    ax = plt.gca()
    plt.axis('off')
    ax.set_xlim([0, W])
    ax.set_ylim([H, 0])
    if save_path is not None:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(save_path, pad_inches=0.0, bbox_inches=extent, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    # return fig2data(fig)


def image_grid(
        images,
        rows=None,
        cols=None,
        show=False,
        fill: bool = False,
        show_axes: bool = False,
        rgb: bool = True,
        save_path: str = None,
        dpi: int = 300
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.
        save_path: image save path
    Returns:
        None
    """
    if len(images) == 0:
        return
    H, W, _ = images[0].shape
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0., "hspace": 0.}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(10, 10. * rows / cols * H / W))
    bleed = 0.
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed), wspace=0., hspace=0.)
    plt.margins(0, 0)

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()

    if save_path is not None:
        fig.savefig(save_path, pad_inches=0.0, dpi=dpi)
    if show:
        plt.show()
    plt.close()
