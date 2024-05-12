from os import path
import cv2 as cv
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from rembg import remove
import time

blur = 21
canny_low = 15
canny_high = 150
min_area = 0.0005
max_area = 0.95
dilate_iter = 10
erode_iter = 10
# mask_color = (0.0, 0.0, 0.0)

params = {
    'blur': blur,
    'canny_low': canny_low,
    'canny_high': canny_high,
    'min_area': min_area,
    'max_area': max_area,
    'dilate_iter': dilate_iter,
    'erode_iter': erode_iter
}

image_path = 'images/me.jpeg'

def remove_bg(image_path=image_path, blur=params['blur'],
              canny_low=params['canny_low'], min_area=params['min_area'],
              max_area=params['max_area'], dilate_iter=params['dilate_iter'],
              erode_iter=params['erode_iter']):
    image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(image_gray, canny_low, canny_high)

    edges = cv.dilate(edges, None)
    edges = cv.erode(edges, None)

    image_area = edges.shape[0] * edges.shape[1]
    min_area = min_area * image_area
    max_area = max_area * image_area

    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    mask = np.zeros(edges.shape, dtype = np.uint8)
    for contour in contours:
        if min_area < cv.contourArea(contour) < max_area:
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
    cv.imshow('final', masked)

    return masked


def d_blur(blur): params['blur'] = blur
def d_canny_low(canny_low): params['canny_low'] = canny_low
def d_canny_high(canny_high): params['canny_high'] = canny_high
def d_min_area(min_area): params['min_area'] = min_area/1000
def d_max_area(max_area): params['max_area'] = max_area/10
def d_dilate_iter(dilate_iter): params['dilate_iter'] = dilate_iter
def d_erode_iter(erode_iter): params['erode_iter'] = erode_iter

cv.namedWindow('controls')
cv.createTrackbar('blur', 'controls', 1, 100, d_blur)
cv.createTrackbar('canny_low', 'controls', 1, 100, d_canny_low)
cv.createTrackbar('canny_high', 'controls', 50, 200, d_canny_high)
cv.createTrackbar('min_area', 'controls', 5, 490, d_min_area)
cv.createTrackbar('max_area', 'controls', 5, 10, d_max_area)
cv.createTrackbar('dilate_iter', 'controls', 1, 25, d_dilate_iter)
cv.createTrackbar('erode_iter', 'controls', 1, 25, d_erode_iter)

while True:
    final = remove_bg(image_path=image_path, blur=params['blur'],
                      canny_low=params['canny_low'], min_area=params['min_area'],
                      max_area=params['max_area'], dilate_iter=params['dilate_iter'],
                      erode_iter=params['erode_iter'])
    time.sleep(1)
    key = cv.waitKey(10)
    if key == ord('s'):
        folder, file = path.split(image_path)
        file_name, ext = path.splitext(file)
        cv.imwrite(path.join(folder, f'{file_name}_foreground{ext}'), final)
        break

cv.destroyAllWindows()
