import numpy as np


def calculate_distance_matrix(coordinates):
    n = len(coordinates)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Calculate Euclidean distance between points i and j
            point1 = np.array(coordinates[i])
            point2 = np.array(coordinates[j])
            distance = np.sqrt(np.sum((point1 - point2) ** 2))
            distance_matrix[i, j] = distance

    return distance_matrix


if __name__ == '__main__':
    # Your coordinates
    coords = [
        [5.001438617706299, 100.513916015625],
        [3.0625576972961426, 101.50936889648438],
        [2.5279293060302734, 102.85059356689453]
    ]

    # Calculate distance matrix
    distances = calculate_distance_matrix(coords)
    print("Distance Matrix:")
    print(distances)