import cv2 as cv
import numpy as np

def grayscale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size=5):
    return cv.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold=50, high_threshold=150):
    return cv.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, top_y=0.6, left_bottom=0.1,
                       left_top=0.45, right_top=0.55,
                       right_bottom=0.9):
    height, width = img.shape[:2] # adds robustness for color images

    polygons = np.array([[
        (int(left_bottom * width), height),
        (int(left_top * width), int(top_y * height)),
        (int(right_top * width), int(top_y * height)),
        (int(right_bottom * width), height)
    ]])

    mask = np.zeros_like(img)
    cv.fillPoly(mask, polygons, 255)
    return cv.bitwise_and(img, mask)

def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = np.reshape(line, 4)
            cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def make_coordinates(img, line_parameters):
    slope, intercept = line_parameters
    y1 = img.shape[0]
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(img, lines):
    if lines is None:
        return []

    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = np.reshape(line, 4)   # works whether line is (1,4) or (4,)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters

        if abs(slope) < 0.5:  # filtering out nearly horizontal lines
            continue

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    averaged_lines = []
    if left_fit:
        left_avg = np.average(left_fit, axis=0)
        averaged_lines.append([make_coordinates(img, left_avg)])
    if right_fit:
        right_avg = np.average(right_fit, axis=0)
        averaged_lines.append([make_coordinates(img, right_avg)])

    return averaged_lines

cap = cv.VideoCapture('testvideo.mp4')

if not cap.isOpened():
    raise RuntimeError("Could not open input video")

fps=cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc(*'mp4v')

out = cv.VideoWriter('lane_detection_output.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = grayscale(frame)
    blur = gaussian_blur(gray)
    edges = canny(blur)

    cropped = region_of_interest(edges)
    lines = cv.HoughLinesP(cropped, 2 , np.pi / 180, 100,
                            np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_img = display_lines(frame, averaged_lines)

    combo = cv.addWeighted(frame, 0.8, line_img, 1, 1)
    out.write(combo)

cap.release()
out.release()
cv.destroyAllWindows()
print("Processing complete. Output video saved!")
