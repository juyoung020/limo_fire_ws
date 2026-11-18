"""Conservative candidate footprint and line-of-sight checks for OccupancyGrid."""
import math


def grid_cell(grid, x, y):
    origin = grid.info.origin
    q = origin.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    dx, dy = x - origin.position.x, y - origin.position.y
    resolution = grid.info.resolution
    if not math.isfinite(resolution) or resolution <= 0:
        return None
    col = math.floor((math.cos(yaw) * dx + math.sin(yaw) * dy) / resolution)
    row = math.floor((-math.sin(yaw) * dx + math.cos(yaw) * dy) / resolution)
    if not (0 <= col < grid.info.width and 0 <= row < grid.info.height):
        return None
    index = row * grid.info.width + col
    return grid.data[index] if index < len(grid.data) else None


def candidate_clear(grid, candidate, target, radius, blocked_cost, target_margin):
    """Check a circumscribed footprint square and the pivot-to-target sight line."""
    step = grid.info.resolution * 0.5
    if not math.isfinite(step) or step <= 0:
        return False

    def free(x, y):
        value = grid_cell(grid, x, y)
        return value is not None and 0 <= value < blocked_cost

    count = max(1, math.ceil(2 * radius / step))
    for i in range(count + 1):
        for j in range(count + 1):
            if not free(candidate.base_x - radius + 2 * radius * i / count,
                        candidate.base_y - radius + 2 * radius * j / count):
                return False
    dx, dy = target[0] - candidate.pivot_x, target[1] - candidate.pivot_y
    length = math.hypot(dx, dy)
    visible_length = max(0.0, length - target_margin)
    count = max(1, math.ceil(visible_length / step))
    for i in range(count + 1):
        fraction = visible_length / length * i / count if length else 0.0
        if not free(candidate.pivot_x + dx * fraction, candidate.pivot_y + dy * fraction):
            return False
    return True
