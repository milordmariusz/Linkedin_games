# import cv2

# def process_image(input_path, output_path):
#     image = cv2.imread(input_path)
#     if image is None:
#         raise ValueError("Invalid image file.")

#     # Add white border
#     bordered_image = cv2.copyMakeBorder(
#         image,
#         50, 50, 50, 50,
#         borderType=cv2.BORDER_CONSTANT,
#         value=[255, 255, 255]
#     )

#     # Resize to 500x500
#     resized_image = cv2.resize(bordered_image, (500, 500))

#     # Save processed image
#     cv2.imwrite(output_path, resized_image)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Constants for configuration
SQUARE_SIZE = 6
COLOR_TOLERANCE = 5
BLUR_SIZE = (5, 5)
CANNY_THRESH1 = 50
CANNY_THRESH2 = 150


def prepare_cropped_image(image, border_size=50):
    """ Preprocess image: add a white border, then blur, edge detection, and crop the largest square contour. """
    
    image_with_border = cv2.copyMakeBorder(
        image, 
        top=border_size, bottom=border_size, left=border_size, right=border_size, 
        borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

    blurred = cv2.GaussianBlur(image_with_border, BLUR_SIZE, 0)
    edges = cv2.Canny(blurred, CANNY_THRESH1, CANNY_THRESH2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour, max_area = None, 0
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                largest_contour = approx

    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image_with_border[y:y + h, x:x + w]
        resized_cropped_image = cv2.resize(cropped_image, (500, 500))
        cv2.imwrite(os.path.join('outputs', f"cropped.jpg"), resized_cropped_image)
    else:
        print("No square detected.")


def find_corners(gray_image):
    """ Find corner points in a grayscale image using Shi-Tomasi method. """
    return cv2.goodFeaturesToTrack(gray_image, maxCorners=100, qualityLevel=0.01, minDistance=10).astype(np.int32)


def sort_corners(corners, row_count):
    """ Sort corner points into rows based on their x, y coordinates. """
    corners = [corner.ravel() for corner in corners]
    corners = sorted(corners, key=lambda point: (point[1], point[0]))
    points_per_row = len(corners) // row_count

    rows = [
        sorted(corners[i * points_per_row:(i + 1) * points_per_row], key=lambda point: point[0])
        for i in range(row_count)
    ]
    return np.array(rows)


def colors_are_similar(color1, color2, tolerance):
    """ Check if two colors are similar within a given tolerance. """
    return np.all(np.abs(np.array(color1) - np.array(color2)) <= tolerance)


def get_mean_color(image, x, y, size):
    """ Get the mean color of a square region around a point (x, y). """
    half_size = size // 2
    region = image[y - half_size:y + half_size, x - half_size:x + half_size]

    if region.shape[0] != size or region.shape[1] != size:
        return np.zeros(3)
    
    return np.mean(region, axis=(0, 1))


def colors_to_matrix(sorted_corners, image):
    """ Convert sorted corner points to a matrix of color indices. """
    color_to_number = {}
    next_color_number = 1

    matrix = []
    matrix_midpoints = []

    for row in range(len(sorted_corners) - 1):
        row_numbers = []
        row_midpoints = []

        for column in range(len(sorted_corners[0]) - 1):
            point_1_x, point_1_y = sorted_corners[row][column]
            point_2_x, point_2_y = sorted_corners[row + 1][column + 1]
            midpoint_x = (point_1_x + point_2_x) // 2
            midpoint_y = (point_1_y + point_2_y) // 2

            row_midpoints.append((midpoint_x, midpoint_y))
            mean_color = get_mean_color(image, midpoint_x, midpoint_y, SQUARE_SIZE)

            found_existing_color = False
            for existing_color in color_to_number:
                if colors_are_similar(mean_color, existing_color, COLOR_TOLERANCE):
                    row_numbers.append(color_to_number[existing_color])
                    found_existing_color = True
                    break

            if not found_existing_color:
                color_to_number[tuple(mean_color)] = next_color_number
                row_numbers.append(next_color_number)
                next_color_number += 1
        
        matrix.append(row_numbers)
        matrix_midpoints.append(row_midpoints)

    return matrix, matrix_midpoints


def find_points(matrix):
    """ Use backtracking to find a valid point arrangement for the matrix. """
    def is_safe(points, x, y, color):
        """ Check if a point is safe to add based on constraints. """
        for px, py, pcolor in points:
            if pcolor == color or px == x or py == y or abs(px - x) <= 1 and abs(py - y) <= 1:
                return False
        return True

    def backtrack(row, points):
        """ Recursively try to find a solution by backtracking. """
        if row == len(matrix):
            return points

        for col in range(len(matrix[row])):
            color = matrix[row][col]
            if is_safe(points, row, col, color):
                points.append((row, col, color))
                result = backtrack(row + 1, points)
                if result:
                    return result
                points.pop()
        return None

    return backtrack(0, [])


def mark_matrix(matrix, points, matrix_midpoints, image):
    """ Mark the matrix with points on the image and return the updated matrix. """
    for x, y, _ in points:
        cv2.circle(image, (matrix_midpoints[x][y][0], matrix_midpoints[x][y][1]), 5, (0, 0, 255), -1)

# Main Execution
def find_queens(file, output_path):
    img = cv2.imread(file)
    prepare_cropped_image(img)

    cropped_path = os.path.join("outputs", "cropped.jpg")
    img = cv2.imread(cropped_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = find_corners(gray)
    row_count = int(math.sqrt(len(corners)))

    sorted_corners = sort_corners(corners, row_count)
    matrix, matrix_midpoints = colors_to_matrix(sorted_corners, img)

    result = find_points(matrix)

    if result:
        mark_matrix(matrix, result, matrix_midpoints, img)
        cv2.imwrite(output_path, img)
    else:
        print("No solution found.")
