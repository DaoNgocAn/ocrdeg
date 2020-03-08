import cv2

import ocrodeg
import numpy as np
import pylab
import scipy.ndimage as ndi

image = cv2.imread("testdata/photo_2020-03-07_21-40-07.jpg")
patch = image


def imshow(img, title='cv2'):
    cv2.imshow(title, img)
    cv2.waitKey(0)


def bounded_gaussian_noise(shape, sigma, maxdelta):
    n, m = shape
    deltas = pylab.rand(2, n, m)
    deltas = ndi.gaussian_filter(deltas, (0, sigma, sigma))
    deltas -= np.amin(deltas)
    deltas /= np.amax(deltas)
    deltas = (2 * deltas - 1) * maxdelta
    return deltas


def distort_with_noise(image, deltas, order=1):
    n, m, _ = image.shape
    xy = np.transpose(np.array(np.meshgrid(
        range(n), range(m))), axes=[0, 2, 1])
    # xy = np.concatenate([xy[:, :, :, None], xy[:, :, :, None], xy[:, :, :, None]], axis=3)
    deltas += xy
    return np.concatenate([ndi.map_coordinates(image[:, :, 0], deltas, order=order, mode="reflect")[:, :, None],
                           ndi.map_coordinates(image[:, :, 1], deltas, order=order, mode="reflect")[:, :, None],
                           ndi.map_coordinates(image[:, :, 2], deltas, order=order, mode="reflect")[:, :, None]],
                          axis=2)


def noise_distort1d(shape, sigma=100.0, magnitude=100.0):
    h, w = shape
    noise = ndi.gaussian_filter(pylab.randn(w), sigma)
    noise *= magnitude / np.amax(abs(noise))
    dys = np.array([noise] * h)
    deltas = np.array([dys, np.zeros((h, w))])
    return deltas


def percent_black(image):
    n = np.prod(image.shape)
    k = np.sum(image < 255 / 2)
    return k * 100.0 / n


def binary_blur(image, sigma, noise=0.0):
    p = percent_black(image)
    blurred = ndi.gaussian_filter(image, sigma) / 255.
    if noise > 0:
        blurred += pylab.randn(*blurred.shape) * noise
    t = np.percentile(blurred, p)
    blurred = np.array(blurred > t, 'f')
    blurred *= 255.

    return blurred.astype(np.uint8)


def make_noise_at_scale(shape, scale):
    h, w = shape
    h0, w0 = int(h / scale + 1), int(w / scale + 1)
    data = pylab.rand(h0, w0)
    with np.warnings.catch_warnings():
        np.warnings.simplefilter("ignore")
        result = ndi.zoom(data, scale)
    return result[:h, :w]


def make_multiscale_noise(shape, scales, weights=None, limits=(0.0, 1.0)):
    if weights is None: weights = [1.0] * len(scales)
    result = make_noise_at_scale(shape, scales[0]) * weights[0]
    for s, w in zip(scales, weights):
        result += make_noise_at_scale(shape, s) * w
    lo, hi = limits
    result -= np.amin(result)
    result /= np.amax(result)
    result *= (hi - lo)
    result += lo
    return result


def make_multiscale_noise_uniform(shape, srange=(1.0, 100.0), nscales=4, limits=(0.0, 1.0)):
    lo, hi = np.log10(srange[0]), np.log10(srange[1])
    scales = np.random.uniform(size=nscales)
    scales = np.add.accumulate(scales)
    scales -= np.amin(scales)
    scales /= np.amax(scales)
    scales *= hi - lo
    scales += lo
    scales = 10 ** scales
    weights = 2.0 * np.random.uniform(size=nscales)
    return make_multiscale_noise(shape, scales, weights=weights, limits=limits)


def autoinvert(image):
    assert np.amin(image) >= 0
    assert np.amax(image) <= 1
    if np.sum(image > 0.9) > np.sum(image < 0.1):
        return 1 - image
    else:
        return image


def random_blobs(shape, blobdensity, size, roughness=2.0):
    from random import randint
    h, w = shape
    numblobs = int(blobdensity * w * h)
    mask = np.zeros((h, w), 'i')
    for i in range(numblobs):
        mask[randint(0, h - 1), randint(0, w - 1)] = 1
    dt = ndi.distance_transform_edt(1 - mask)
    mask = np.array(dt < size, 'f')
    mask = ndi.gaussian_filter(mask, size / (2 * roughness))
    mask -= np.amin(mask)
    mask /= np.amax(mask)
    noise = pylab.rand(h, w)
    noise = ndi.gaussian_filter(noise, size / (2 * roughness))
    noise -= np.amin(noise)
    noise /= np.amax(noise)
    return np.array(mask * noise > 0.5, 'f')


def random_blotches(image, fgblobs, bgblobs, fgscale=10, bgscale=10):
    fg = random_blobs(image.shape, fgblobs, fgscale)
    bg = random_blobs(image.shape, bgblobs, bgscale)
    return np.minimum(np.maximum(image, fg), 1 - bg)

def make_fiber(l, a, stepsize=0.5):
    angles = np.random.standard_cauchy(l) * a
    angles[0] += 2 * np.pi * pylab.rand()
    angles = np.add.accumulate(angles)
    coss = np.add.accumulate(np.cos(angles) * stepsize)
    sins = np.add.accumulate(np.sin(angles) * stepsize)
    return np.array([coss, sins]).transpose(1, 0)

def make_fibrous_image(shape, nfibers=300, l=300, a=0.2, stepsize=0.5, limits=(0.1, 1.0), blur=1.0):
    h, w = shape
    lo, hi = limits
    result = np.zeros(shape)
    for i in range(nfibers):
        v = pylab.rand() * (hi - lo) + lo
        fiber = make_fiber(l, a, stepsize=stepsize)
        y, x = np.randint(0, h - 1), np.randint(0, w - 1)
        fiber[:, 0] += y
        fiber[:, 0] = np.clip(fiber[:, 0], 0, h - .1)
        fiber[:, 1] += x
        fiber[:, 1] = np.clip(fiber[:, 1], 0, w - .1)
        for y, x in fiber:
            result[int(y), int(x)] = v
    result = ndi.gaussian_filter(result, blur)
    result -= np.amin(result)
    result /= np.amax(result)
    result *= (hi - lo)
    result += lo
    return result


def printlike_multiscale(image, blur=1.0, blotches=5e-5):
    selector = autoinvert(image)
    selector = random_blotches(selector, 3 * blotches, blotches)
    paper = make_multiscale_noise_uniform(image.shape, limits=(0.5, 1.0))
    ink = make_multiscale_noise_uniform(image.shape, limits=(0.0, 0.5))
    blurred = ndi.gaussian_filter(selector, blur)
    printed = blurred * ink + (1 - blurred) * paper
    return printed

def printlike_fibrous(image, blur=1.0, blotches=5e-5):
    selector = autoinvert(image)
    selector = random_blotches(selector, 3 * blotches, blotches)
    paper = make_multiscale_noise(image.shape, [1.0, 5.0, 10.0, 50.0], weights=[1.0, 0.3, 0.5, 0.3], limits=(0.7, 1.0))
    paper -= make_fibrous_image(image.shape, 300, 500, 0.01, limits=(0.0, 0.25), blur=0.5)
    ink = make_multiscale_noise(image.shape, [1.0, 5.0, 10.0, 50.0], limits=(0.0, 0.5))
    blurred = ndi.gaussian_filter(selector, blur)
    printed = blurred * ink + (1 - blurred) * paper
    return printed



# RANDOM DISTORTIONS Done
# for i, sigma in enumerate([1.0, 2.0, 5.0, 20.0]):
#     noise = bounded_gaussian_noise(image.shape[:2], sigma, 5.0)
#     distorted = distort_with_noise(image, noise)
#     h, w, _ = image.shape
#     imshow(distorted[h // 2 - 200:h // 2 + 200, w // 3 - 200:w // 3 + 200], f'{i}')

# RULED SURFACE DISTORTIONS : DONE
# for i, mag in enumerate([5.0, 20.0, 100.0, 200.0]):
#     noise = noise_distort1d(image.shape[:2], magnitude=mag)
#     distorted = distort_with_noise(image, noise)
#     h, w, _ = image.shape
#     imshow(distorted[:1500], f'{i}')

# lens blur
# for i, s in enumerate([0, 1, 2, 4]):
#     blurred = ndi.gaussian_filter(patch, s)
#     imshow(blurred)


# lens blur with thresholding
# for i, s in enumerate([0, 1, 2, 4]):
#     blurred = ndi.gaussian_filter(patch, s)
#     blurred[blurred > (255 / 2)] = 255
#     thresholded = blurred
#     imshow(thresholded)


# lens blur with thresholding and constant black pixels
# for i, s in enumerate([0.0, 1.0, 2.0, 4.0]):
#     blurred = binary_blur(patch, s)
#     imshow(blurred)


# lens blur with additive noise
# for i, s in enumerate([0.0, 0.1, 0.2, 0.3]):
#     blurred = binary_blur(patch, 0.5, noise=s)
#     imshow(blurred)

#  Failed
# for i in range(100):
#     noisy = make_multiscale_noise_uniform((patch.shape[0], patch.shape[1]))
#     distorted = patch[:,:,0] + noisy
#     imshow(distorted)

# while True:
#     imshow(printlike_multiscale(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)/255.))
#     imshow(np.concatenate([printlike_multiscale(patch[:, :, 0] / 255.)[:, :, None],
#                            printlike_multiscale(patch[:, :, 1] / 255.)[:, :, None],
#                            printlike_multiscale(patch[:, :, 2] / 255.)[:, :, None]], axis=2))


imshow(ocrodeg.printlike_fibrous(patch[:,:,0]/255.))