import numpy as np

def create_pointcloud(color_img, depth_img, semantic_map, num_points, original_img_size, offset, fx, fy, cx, cy):
    h, w = depth_img.shape[:2]
    granularity = int(np.sqrt((w * h) / num_points))
    if granularity < 1:
        granularity = 1

    num_points = min(num_points, np.count_nonzero(depth_img))

    points = np.zeros((num_points, 7))
    i = 0

    for x in range(0, w, granularity):
        for y in range(0, h, granularity):
            depth = depth_img[y][x]
            if depth == 0:
                continue

            Z = depth
            X = (x - cx) * Z / fx
            Y = (y - cy) * Z / fy

            color = color_img[y][x]
            semantic_id = semantic_map[y][x]

            points[i] = [
                X, Z, -Y,
                int(color[0]), int(color[1]), int(color[2]),
                int(semantic_id)
            ]

            i += 1
            if i == num_points:
                break
        else:
            continue
        break

    points.resize((i, 7))
    return points


def create_pointcloud_adaptive(color_img, depth_img, semantic_map, num_points, original_img_size, offset):
    h, w = depth_img.shape[:2]
    hp, wp = h / original_img_size[1], w / original_img_size[0]
    step = max(1, int(np.count_nonzero(depth_img) / num_points))

    points = np.zeros((num_points, 7))
    points_found = 0
    points_stored = 0

    non_zero_indices = np.nonzero(depth_img)
    non_zero_indices = zip(non_zero_indices[0], non_zero_indices[1])

    for y, x in list(non_zero_indices):
        depth = depth_img[y][x]
        points_found += 1

        if (points_found - 1) % step != 0:
            continue

        x_pos = ((float(x) + offset[0]) / original_img_size[0] - .5) * depth
        y_pos = -((float(y) + offset[1]) / original_img_size[1] - .5) * depth * (
            original_img_size[1] / original_img_size[0]
        )

        color = color_img[y][x]
        semantic_id = semantic_map[y][x]

        points[points_stored] = [
            x_pos, depth, y_pos,
            int(color[0]), int(color[1]), int(color[2]),
            int(semantic_id)
        ]

        points_stored += 1
        if points_stored == num_points:
            break

    points.resize((points_stored, 7))
    return points


def create_balanced_semantic_pointcloud(color_img, depth_img, semantic_map, num_points, fx, fy, cx, cy):
    class_ids = [c for c in np.unique(semantic_map) if c != 0]

    if len(class_ids) == 0:
        return np.zeros((0, 7))

    points_per_class = max(1, num_points // len(class_ids))
    all_points = []

    for class_id in class_ids:
        points = create_pointcloud_from_mask_random(
            color_img,
            depth_img,
            semantic_map,
            class_id,
            points_per_class,
            fx,
            fy,
            cx,
            cy
        )

        if len(points) > 0:
            all_points.append(points)

    if len(all_points) == 0:
        return np.zeros((0, 7))

    return np.vstack(all_points)


def create_pointcloud_from_mask_random(color_img, depth_img, semantic_map, class_id, num_points, fx, fy, cx, cy):
    ys, xs = np.where((semantic_map == class_id) & (depth_img > 0))

    if len(xs) == 0:
        return np.zeros((0, 7))

    n = min(num_points, len(xs))
    idx = np.random.choice(len(xs), size=n, replace=False)

    xs = xs[idx]
    ys = ys[idx]

    points = np.zeros((n, 7))

    for i, (x, y) in enumerate(zip(xs, ys)):
        Z = depth_img[y, x]
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy

        color = color_img[y, x]
        semantic_id = semantic_map[y, x]

        points[i] = [
            X, Z, -Y,
            int(color[0]), int(color[1]), int(color[2]),
            int(semantic_id)
        ]

    return points


def create_area_aware_semantic_pointcloud(color_img, depth_img, semantic_map, num_points, fx, fy, cx, cy):
    """
    Area-aware semantic point cloud sampling.

    Goal:
    - Large classes get more points than small classes.
    - Small classes still remain visible.
    - Huge classes like wall/floor cannot dominate everything.
    - No manual class weights needed.
    """

    class_ids = [c for c in np.unique(semantic_map) if c != 0]

    if len(class_ids) == 0:
        return np.zeros((0, 7))

    min_points_per_class = max(100, int(0.03 * num_points))
    max_fraction_per_class = 0.45

    max_points_per_class = int(num_points * max_fraction_per_class)

    class_pixel_counts = {}
    for class_id in class_ids:
        count = np.count_nonzero((semantic_map == class_id) & (depth_img > 0))
        if count > 0:
            class_pixel_counts[class_id] = count

    if len(class_pixel_counts) == 0:
        return np.zeros((0, 7))

    class_scores = {
        class_id: np.sqrt(pixel_count)
        for class_id, pixel_count in class_pixel_counts.items()
    }

    total_score = sum(class_scores.values())

    all_points = []

    for class_id, score in class_scores.items():
        raw_budget = int(num_points * (score / total_score))

        class_budget = max(min_points_per_class, raw_budget)
        class_budget = min(max_points_per_class, class_budget)
        class_budget = min(class_budget, class_pixel_counts[class_id])

        points = create_pointcloud_from_mask_grid(
            color_img,
            depth_img,
            semantic_map,
            class_id,
            class_budget,
            fx,
            fy,
            cx,
            cy
        )

        if len(points) > 0:
            all_points.append(points)

    if len(all_points) == 0:
        return np.zeros((0, 7))

    points = np.vstack(all_points)

    if len(points) > num_points:
        idx = np.random.choice(len(points), size=num_points, replace=False)
        points = points[idx]

    return points


def create_pointcloud_from_mask_grid(color_img, depth_img, semantic_map, class_id, num_points, fx, fy, cx, cy):
    """
    Stable grid-like sampling inside one semantic class.
    This avoids random flickering and gives better visual coverage.
    """

    ys, xs = np.where((semantic_map == class_id) & (depth_img > 0))

    if len(xs) == 0:
        return np.zeros((0, 7))

    n_available = len(xs)
    n = min(num_points, n_available)

    order = np.lexsort((xs, ys))
    xs = xs[order]
    ys = ys[order]

    if n_available > n:
        idx = np.linspace(0, n_available - 1, n, dtype=np.int32)
        xs = xs[idx]
        ys = ys[idx]

    points = np.zeros((len(xs), 7))

    for i, (x, y) in enumerate(zip(xs, ys)):
        Z = depth_img[y, x]
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy

        color = color_img[y, x]
        semantic_id = semantic_map[y, x]

        points[i] = [
            X, Z, -Y,
            int(color[0]), int(color[1]), int(color[2]),
            int(semantic_id)
        ]

    return points