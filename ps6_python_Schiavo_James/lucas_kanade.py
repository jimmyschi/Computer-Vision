import cv2 as cv 
import numpy as np
import multiprocess as mp
from matplotlib import pyplot as plt 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def lucas_kanade(img1,img2,window_size=None):
    if window_size == None:
        window_sz = min(max(3, min(img1.shape[:2]) / 4), 50)
        window_sz = int(2 * (window_sz // 2) + 1)
        window_size = (window_sz, window_sz)
    t0, t1 = img1.astype(np.float32), img2.astype(np.float32)
    ix, iy = np.zeros(t0.shape, dtype=np.float32), np.zeros(t0.shape, dtype=np.float32)
    ix[1:-1, 1:-1, ...] = t0[1:-1, 2:, ...] - t0[1:-1, :-2, ...]
    iy[1:-1, 1:-1, ...] = t0[2:, 1:-1, ...] - t0[:-2, 1:-1, ...]
    it = (t0 - t1).astype(np.float32)
    grad_products = [ix * ix, ix * iy, iy * iy, ix * it, iy * it]
    grad_products = [np.sum(grad_prod, axis=2) for grad_prod in grad_products] if len(img1.shape) == 3 else grad_products
    wgps = [cv.GaussianBlur(grad_prod, window_size, 0) for grad_prod in grad_products]
    wgps = np.concatenate([wgp[:, :, np.newaxis] for wgp in wgps], axis=2)
    flow = np.zeros(img1.shape[:2] + (2,), dtype=np.float32)
    (w_h, w_l) = map(lambda x: (x - 1) // 2, window_size)
    with mp.Pool(10) as pool:
        def i_slice(i_var):
            uv_slice = np.zeros((img1.shape[1], 2))
            for j in range(w_l, img1.shape[1] - w_l):
                a = [[wgps[i_var, j, 0], wgps[i_var, j, 1]], [wgps[i_var, j, 1], wgps[i_var, j, 2]]]
                b = [[-wgps[i_var, j, 3]], [-wgps[i_var, j, 4]]]
                uv, _, rank, _ = np.linalg.lstsq(a, b, rcond=0.2)
                u,v = uv.flatten() if rank == 2 else [0, 0]
                uv_slice[j, :] = [u, v]
            return uv_slice
        i_range = range(w_h, img1.shape[0] - w_h)
        slices = pool.map(i_slice, list(i_range))
        for i, slice_i in zip(i_range, slices):
            flow[i, :, :] = slice_i
    return -flow



def draw_flow(img,flow,gap=None):
    img = img.copy()
    if len(img.shape) == 2:
        im = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    gap = [gap, flow.shape[0] // 30][gap is None]
    gap = max(gap, 1)
    x = np.arange(0, flow.shape[1], 1)
    y = np.arange(0, flow.shape[0], 1)
    x, y = np.meshgrid(x, y)
    figure = plt.figure()
    axes = figure.add_axes([0, 0, 1, 1], frameon=False)
    figure.patch.set_alpha(0)
    axes.patch.set_alpha(0)
    canvas = FigureCanvas(figure)
    plt.quiver(x[::gap, ::gap], y[::-gap, ::-gap], flow[::gap, ::gap, 0], -flow[::gap, ::gap, 1], color='red')
    axes.axis('off')
    axes.margins(0)
    canvas.draw()
    image = np.frombuffer(figure.canvas.tostring_argb(), dtype=np.uint8)
    image = image.reshape(figure.canvas.get_width_height()[::-1] + (4,))
    arrows = cv.resize(image, dsize=flow.shape[1::-1])
    alpha = arrows[:, :, 0]
    for i in range(3):
        img[:, :, i] = (alpha / 255.0) * arrows[:, :, 1 + i] + ((255.0 - alpha) / 255.0) * img[:, :, i]
    return img


