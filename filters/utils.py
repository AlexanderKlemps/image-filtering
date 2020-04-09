import numpy as np
import cv2


def put_text(image, text, pos, color=(255, 255, 255), size=1):
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 1, cv2.LINE_AA)


def resize(image, scale=0.5):
    return cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


def gradient_x(image):
    kernel_x = 0.5 * np.array([[0, 0, 0],
                               [1, 0, -1],
                               [0, 0, 0]], dtype=np.float32)

    dx = cv2.filter2D(image, cv2.CV_32F, kernel_x)

    return dx


def gradient_y(image):
    kernel_y = 0.5 * np.array([[0, 1, 0],
                               [0, 0, 0],
                               [0, -1, 0]], dtype=np.float32)

    dy = cv2.filter2D(image, cv2.CV_32F, kernel_y)

    return dy


def gradient(image):
    dx, dy = gradient_x(image), gradient_y(image)

    return np.dstack((dx, dy))


def divergence(u):
    u1 = u[..., 0]
    u2 = u[..., 1]

    dx = gradient_x(u1)
    dy = gradient_y(u2)

    return dx + dy


def laplace(image, standard=True):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    if not standard:
        s = 1 / 3
        kernel = s * kernel + (1 - s) * np.array([[1, 0, 1],
                                                  [0, -4, 0],
                                                  [1, 0, 1]], dtype=np.float32)

    return cv2.filter2D(image, cv2.CV_32F, kernel)
