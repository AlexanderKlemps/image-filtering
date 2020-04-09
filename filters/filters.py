import numpy as np
from .utils import gradient, laplace, divergence, resize
import cv2
import time


class FilterStack(object):

    def __init__(self, filters, scale=-1.0, verbosity=0):
        self.filters = filters
        self.verbosity = verbosity
        self.scale_factor = scale
        self.scale = True
        if (1 <= self.scale_factor) or (self.scale_factor <= 0):
            self.scale = False

    def apply(self, image):
        filtered_image = image.copy()
        if self.scale:
            filtered_image = resize(filtered_image, self.scale_factor)
        for image_filter in self.filters:
            filtered_image = image_filter.process(filtered_image)

        if self.scale:
            filtered_image = resize(filtered_image, 1 / self.scale_factor)

        if len(image.shape) > 2 and len(filtered_image.shape) == 2:
            filtered_image = np.dstack(tuple([filtered_image] * image.shape[2]))

        return filtered_image


class BaseFilter(object):

    def __init__(self, verbosity=0, **kwargs):
        self.verbosity = verbosity
        self.name = "BaseFilter"
        for key, value in kwargs.items():
            setattr(self, key, value)

    def process(self, image):
        t0 = time.time()
        img = image.copy()
        img = img.astype(np.float32)
        processed = self._process(img)
        if self.verbosity > 0:
            diff = time.time() - t0
            print("{} needed {:.2f} seconds for processing".format(self.name, diff))
        return processed

    def _process(self, image):
        return image


class OsmosisFilter(BaseFilter):
    """
        Implementation of Linear Osmosis Filter by J. Weickert et al.
        See https://www.mia.uni-saarland.de/Publications/vogel-ssvm13.pdf for more information.
    """

    def __init__(self, verbosity=0, **kwargs):
        """
        :param verbosity: Verbosity parameter to control debugging.
        :param kwargs: Parameters that can be set:
            - n_iter: number of iterations applied to the image
            - tau: time step size used during iteration
        """
        self.name = "OsmosisFilter"
        self.n_iter = 5
        self.tau = 0.1
        super(OsmosisFilter, self).__init__(verbosity=verbosity, **kwargs)
        self.parameters = {"n_iter": self.n_iter, "tau": self.tau}
        self.max_values = {"n_iter": 100., "tau": 2.}

    def blur(self, image):
        return cv2.GaussianBlur(image, (3, 3), 0)

    def _process(self, img):
        """
        :param img: Image array the filter should be applied to.
                    Should be of type float and either colored or gray scaled.
        :return: Filtered image
        """

        tau = self.parameters["tau"]
        n_iter = int(self.parameters["n_iter"])

        # the pixel values have to be strictly positive to apply method
        # to assure that, shift zero valued pixels
        if len(img.shape) > 2:
            for i in range(img.shape[2]):
                img[..., i][img[..., i] == 0] = 1
        else:
            img[img == 0] = 1

        # shifting some pixels to 1 causes discontinuities
        # to fix those, apply blur
        img = self.blur(img)
        avg_grey_value = np.mean(np.mean(img, axis=0), axis=0)
        u = np.zeros_like(img, dtype=np.float32) + avg_grey_value

        D = gradient(img)
        shape = tuple(list(D.shape[:2]) + [D.shape[2] // 3] + [3])
        D = D.reshape(shape).transpose((0, 1, 3, 2))

        def f(u):
            sub = np.dstack((u / img, u / img)).reshape(shape).transpose((0, 1, 3, 2))
            f = laplace(u) - divergence(D * sub)
            return f

        for i in range(n_iter):
            u = u + tau * f(u)
            u = np.clip(u, 0, 255)
            # u = u + 0.5 * self.tau * (f(u) + f(u_))

        return cv2.convertScaleAbs(u).astype(np.uint8)


class CoherenceShockFilter(BaseFilter):

    def __init__(self, verbosity=0, **kwargs):
        self.name = "CoherenceShockFilter"
        self.sigma = 11
        self.str_sigma = 11
        self.blend = 0.5
        self.n_iter = 4
        super(CoherenceShockFilter, self).__init__(verbosity=verbosity, **kwargs)
        self.parameters = {"n_iter": self.n_iter, "sigma": self.sigma, "str_sigma": self.str_sigma,
                           "blend": self.blend}
        self.max_values = {"n_iter": 100., "sigma": 15., "str_sigma": 15., "blend": 1.}

    def _process(self, img):
        n_iter = int(self.parameters["n_iter"])
        sigma = 2 * int(self.parameters["sigma"]) + 1
        str_sigma = 2 * int(self.parameters["str_sigma"]) + 1
        blend = self.parameters["blend"]

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        h, w = gray.shape

        for i in range(n_iter):
            eigen = cv2.cornerEigenValsAndVecs(gray, str_sigma, 3)
            eigen = eigen.reshape(h, w, 3, 2)
            x, y = eigen[:, :, 1, 0], eigen[:, :, 1, 1]

            gxx = cv2.Sobel(gray, cv2.CV_64F, 2, 0, ksize=sigma)
            gxy = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=sigma)
            gyy = cv2.Sobel(gray, cv2.CV_64F, 0, 2, ksize=sigma)
            gvv = x * x * gxx + 2 * x * y * gxy + y * y * gyy
            m = gvv < 0

            ero = cv2.erode(img, None)
            dil = cv2.dilate(img, None)
            img1 = ero
            img1[m] = dil[m]
            img = np.uint8(img * (1.0 - blend) + img1 * blend)

        return img


class PeronaMalikFilter(BaseFilter):
    """
    Implementation of the isotropic Perona-Malik-Filter.
    See http://image.diku.dk/imagecanon/material/PeronaMalik1990.pdf for more information.
    """

    def __init__(self, verbosity=0, **kwargs):
        self.name = "PeronaMalikFilter"
        self.lamb = 2.0
        self.tau = 0.1
        self.n_iter = 5
        super(PeronaMalikFilter, self).__init__(verbosity=verbosity, **kwargs)
        self.parameters = {"n_iter": self.n_iter, "tau": self.tau, "lamb": self.lamb}
        self.max_values = {"n_iter": 100., "tau": 2., "lamb": 10.}

    def _process(self, img):
        """
        :param img: Image array the filter should be applied to.
                            Should be of type float and either colored or gray scaled.
        :return: Filtered image
        """

        lamb = self.parameters["lamb"]
        n_iter = int(self.parameters["n_iter"])
        tau = self.parameters["tau"]

        self.diffusity = lambda u: 1 / (1 + (u / lamb ** 2))

        def f(img):
            d = gradient(img)
            shape = tuple(list(d.shape[:2]) + [d.shape[2] // 3] + [3])
            d = d.reshape(shape).transpose((0, 1, 3, 2))
            g = self.diffusity(np.linalg.norm(np.square(np.sum(d, axis=3)), axis=2))
            G = np.dstack([g] * 6).reshape(shape).transpose((0, 1, 3, 2))

            f = divergence(G * d)

            return f

        for i in range(n_iter):
            img = img + tau * f(img)
            # img = img + 0.5 * self.tau * (f(img) + f(img_))
            img = np.clip(img, 0, 255)

        return img.astype(np.uint8)


class SobelFilter(BaseFilter):

    def __init__(self, verbosity=0, **kwargs):
        self.name = "SobelFilter"
        self.sigma = 5
        super(SobelFilter, self).__init__(verbosity=verbosity, **kwargs)

    def _process(self, image):
        img = image
        if len(image.shape) > 2:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sobel_x = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=self.sigma)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=self.sigma)

        sobel_x = np.absolute(sobel_x)
        sobel_x = cv2.convertScaleAbs(sobel_x)
        sobel_y = cv2.convertScaleAbs(sobel_y)

        sobel = cv2.bitwise_or(sobel_x, sobel_y)

        return sobel


class GaussianFilter(BaseFilter):

    def __init__(self, verbosity=0, **kwargs):
        self.name = "GaussianFilter"
        self.sigma = 5
        super(GaussianFilter, self).__init__(verbosity=verbosity, **kwargs)

    def _process(self, image):
        img = cv2.GaussianBlur(image, (self.sigma, self.sigma), 0)
        return img


class CannyFilter(BaseFilter):

    def __init__(self, verbosity=0, **kwargs):
        self.name = "CannyFilter"
        self.sigma = 5
        super(CannyFilter, self).__init__(verbosity=verbosity, **kwargs)

    def _process(self, image):
        img = cv2.Canny(image, 50, 80, L2gradient=True)

        return img


class ThreshFilter(BaseFilter):

    def __init__(self, verbosity=0, **kwargs):
        self.name = "ThreshFilter"
        self.thresh = 20
        super(ThreshFilter, self).__init__(verbosity=verbosity, **kwargs)

    def _process(self, image):
        img = image
        if len(image.shape) > 2:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(img, self.thresh, 255, cv2.THRESH_BINARY)
        return thresh
