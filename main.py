from os import path
import cv2 as cv
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

image_path = 'images/nezuko.jpg'
image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
final = image

blur = 21
min_area = 0.0005
max_area = 0.95
canny_low = 15
canny_high = 150
dilate_iter = 10
erode_iter = 10

modify = False
draw = None
radius = 5

params = {
    'min_area': min_area,
    'max_area': max_area,
    'canny_low': canny_low,
    'canny_high': canny_high,
    'dilate_iter': dilate_iter,
    'erode_iter': erode_iter
}

def remove_bg(image=image, canny_low=params['canny_low'], canny_high=params['canny_high'],
              min_area=params['min_area'], max_area=params['max_area'],
              dilate_iter=params['dilate_iter'], erode_iter=params['erode_iter']):

    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(image_gray, canny_low, canny_high)
    edges = cv.dilate(edges, None)
    edges = cv.erode(edges, None)

    image_area = edges.shape[0] * edges.shape[1]
    min_contour = min_area * image_area
    max_contour = max_area * image_area

    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    mask = np.zeros(edges.shape, dtype = np.uint8)
    for contour in contours:
        if min_contour < cv.contourArea(contour) < max_contour:
            contour = contour.reshape(-1, 2)
            mask = cv.fillConvexPoly(mask, contour, (255))

    mask = cv.dilate(mask, None, iterations=dilate_iter)
    mask = cv.erode(mask, None, iterations=erode_iter)
    mask = cv.GaussianBlur(mask, (blur, blur), 0)

    mask_stack = mask.astype('float32') / 255.0
    mask_stack = np.repeat(mask_stack[:, :, np.newaxis], 3, axis=2)
    frame = image.astype('float32') / 255.0

    masked = (mask_stack * frame) + ((1-mask_stack) * (0.0, 0.0, 0.0))
    masked = (masked * 255).astype('uint8')

    return masked


def d_min_area(min_area):
    global final, image
    params['min_area'] = min_area/1000
    final = remove_bg(image, **params)

def d_max_area(max_area):
    global final, image
    params['max_area'] = max_area/10
    final = remove_bg(image, **params)

def d_canny_low(canny_low):
    global final, image
    params['canny_low'] = canny_low
    final = remove_bg(image, **params)

def d_canny_high(canny_high):
    global final, image
    params['canny_high'] = canny_high
    final = remove_bg(image, **params)

def d_dilate_iter(dilate_iter):
    global final, image
    params['dilate_iter'] = dilate_iter
    final = remove_bg(image, **params)

def d_erode_iter(erode_iter):
    global final, image
    params['erode_iter'] = erode_iter
    final = remove_bg(image, **params)

final = remove_bg(image=image, **params)

cv.namedWindow('controls')
cv.createTrackbar('min_area', 'controls', 5, 490, d_min_area)
cv.createTrackbar('max_area', 'controls', 5, 10, d_max_area)
cv.createTrackbar('canny_low', 'controls', 1, 100, d_canny_low)
cv.createTrackbar('canny_high', 'controls', 50, 200, d_canny_high)
cv.createTrackbar('dilate_iter', 'controls', 1, 25, d_dilate_iter)
cv.createTrackbar('erode_iter', 'controls', 1, 25, d_erode_iter)

while True:
    cv.imshow('result', final)
    key = cv.waitKey(10)
    if key == ord('n'):
        cv.destroyWindow('controls')
        break

def correct(event, x, y, flags, param):
    global modify, radius, draw, image, final
    if event == cv.EVENT_MOUSEWHEEL:
        if flags > 0: radius += 5
        else: radius -= 5
    if event == cv.EVENT_LBUTTONDOWN:
        modify = True
        draw = 'erase'
    if (draw == 'draw') and event == cv.EVENT_MOUSEMOVE:
        if modify:
            r = radius
            final[y - r:y + r, x - r:x + r] \
                = image[y - r:y + r, x - r:x + r]
            # cv.circle(image, (x, y), radius, (255, 0, 0), -1)
    if (draw == 'erase') and event == cv.EVENT_MOUSEMOVE:
        if modify:
            cv.circle(image, (x, y), radius, (0, 0, 0), -1)
    if event == cv.EVENT_LBUTTONUP:
        modify = False
        draw = None
    if event == cv.EVENT_RBUTTONDOWN:
        modify = True
        draw = 'draw'
    if event == cv.EVENT_RBUTTONUP:
        modify = False
        draw = None


cv.namedWindow('result')
cv.setMouseCallback('result', correct)

while True:
    cv.imshow('result', final)
    key = cv.waitKey(10)
    if key == ord('s'):
        folder, file = path.split(image_path)
        file_name, ext = path.splitext(file)
        cv.imwrite(path.join(folder, f'{file_name}_foreground{ext}'), final)
        break

cv.destroyAllWindows()
