from cv2 import approxPolyDP, boundingRect, arcLength

class ShapeDetector(object):
    def __init__(self):
        pass


    def detect_box(self, contour):
        
        peri = arcLength(contour, True)
        approx = approxPolyDP(contour, 0.04 * peri, True)

        if len(approx) == 4:
            (x, y, w, h) = boundingRect(approx)
            aspect_ratio = w / float(h)
            print(x, y)
            if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
                return True


