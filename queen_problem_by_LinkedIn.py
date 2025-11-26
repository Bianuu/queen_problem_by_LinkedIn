import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from sklearn.cluster import KMeans


def detect_grid_and_regions(image_path, k_regions=None):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Nu pot încărca imaginea!")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold pentru liniile negre
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # detectare linii verticale
    vertical_projection = np.sum(thresh, axis=0)
    vertical_lines = np.where(vertical_projection > (0.3 * h * 255))[0]

    v_lines = []
    prev = -10
    for x in vertical_lines:
        if x - prev > 2:
            v_lines.append(x)
        prev = x

    # detectare linii orizontale
    horizontal_projection = np.sum(thresh, axis=1)
    horizontal_lines = np.where(horizontal_projection > (0.3 * w * 255))[0]

    h_lines = []
    prev = -10
    for y in horizontal_lines:
        if y - prev > 2:
            h_lines.append(y)
        prev = y

    # determinăm N
    N = len(v_lines) - 1

    # extragem culoarea din centrul fiecărei celule
    cell_colors = []
    for i in range(N):
        for j in range(N):
            y1, y2 = h_lines[i], h_lines[i + 1]
            x1, x2 = v_lines[j], v_lines[j + 1]
            cy = (y1 + y2) // 2
            cx = (x1 + x2) // 2
            cell_colors.append(img[cy, cx])

    cell_colors = np.array(cell_colors)

    # estimăm numărul de culori reale
    if k_regions is None:
        uniq = np.unique(cell_colors.reshape(-1, 3), axis=0)
        k_regions = min(len(uniq), 20)

    # clustering de culori
    kmeans = KMeans(n_clusters=k_regions, n_init=10).fit(cell_colors)
    labels = kmeans.labels_

    # construim matricea NxN de regiuni
    region_matrix = labels.reshape(N, N) + 1
    region_matrix = region_matrix.astype(int).tolist()  # <- int normal, nu np.int32

    return region_matrix


def solve_star_puzzle(region_matrix):
    N = len(region_matrix)

    # extragem toate culorile distincte
    all_colors = sorted({region_matrix[r][c] for r in range(N) for c in range(N)})
    used_color = {c: False for c in all_colors}

    used_row = [False] * N
    used_col = [False] * N

    solution = [[region_matrix[r][c] for c in range(N)] for r in range(N)]

    neigh = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1), (0, 1),
             (1, -1), (1, 0), (1, 1)]

    def can_place(r, c):
        for dr, dc in neigh:
            rr, cc = r + dr, c + dc
            if 0 <= rr < N and 0 <= cc < N:
                if solution[rr][cc] == '*':
                    return False
        return True

    positions = [(r, c) for r in range(N) for c in range(N)]

    def backtrack(i):
        if i == len(positions):
            return all(used_color.values())

        r, c = positions[i]
        color = region_matrix[r][c]

        # Dacă zona are deja steluță, trecem mai departe
        if used_color[color]:
            return backtrack(i + 1)

        # Încercăm să plasăm steaua
        if (not used_row[r] and
                not used_col[c] and
                not used_color[color] and
                can_place(r, c)):

            solution[r][c] = '*'
            used_row[r] = True
            used_col[c] = True
            used_color[color] = True

            if backtrack(i + 1):
                return True

            # revert
            solution[r][c] = color
            used_row[r] = False
            used_col[c] = False
            used_color[color] = False

        # Fără stea aici
        return backtrack(i + 1)

    if backtrack(0):
        return solution

    return None


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Alege imaginea grilei",
        filetypes=[("Imagini", "*.png *.jpg *.jpeg *.bmp *.webp"), ("Toate fișierele", "*.*")]
    )

    if not file_path:
        print("Nu a fost selectat niciun fișier.")
        exit()

    # 1) detectăm matricea de regiuni
    matrix = detect_grid_and_regions(file_path)

    print("Matricea de regiuni:")
    print("[")
    for row in matrix:
        print(" ", row, ",")
    print("]")

    # 2) rezolvăm puzzle-ul cu stele
    sol = solve_star_puzzle(matrix)

    print("\nSolutia cu stele:")
    if sol:
        for row in sol:
            print(row)
    else:
        print("Nu exista solutie.")
