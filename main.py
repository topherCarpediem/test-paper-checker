import time

import cv2
from imutils.perspective import four_point_transform
from imutils import is_cv2
from detector.shape.shape_detector import ShapeDetector

orig_image = cv2.imread('test.png')
gray_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
edged_image = cv2.Canny(blurred_image, 75, 200)
thresh_image = cv2.threshold(blurred_image, 60, 255, cv2.THRESH_BINARY)[1]



contours = cv2.findContours(
    edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if is_cv2() else contours[1]
paperContour = None

if len(contours) > 0:

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            cv2.drawContours(orig_image, [contour], -1, (0, 255, 0), 2)
            paperContour = approx
            break

cv2.imshow('orig', orig_image)
warped_image = four_point_transform(gray_image, paperContour.reshape(4,2))

thresh = cv2.threshold(warped_image, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cv2.imshow('thresh', thresh)
# sd = ShapeDetector()

# for contour in contours:
#     cv2.drawContours(orig_image, [contour], -1, (0, 255, 0), 2)
#     if sd.detect_box(contour):
#         print('box detected')




while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
