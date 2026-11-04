"""Keep mapped fire landmarks until ObjectMapper explicitly removes them."""
import math


MARKER_ADD = 0
MARKER_DELETE = 2
MARKER_DELETEALL = 3


def update_fire_marker_cache(cache, markers, namespace, marker_part, transform):
    """Apply a MarkerArray update to ``id // 10 -> (x, y)`` landmarks.

    ``transform(marker)`` returns the marker position in the shared map frame,
    or ``None`` when its transform is not available. Markers are persistent map
    records, so an unrelated or empty update does not erase known landmarks.
    Only explicit DELETE/DELETEALL actions remove them.
    """
    changed = False
    for marker in markers:
        if marker.action == MARKER_DELETEALL:
            cache.clear()
            changed = True
            continue
        if marker.ns != namespace or marker.id % 10 != marker_part:
            continue

        landmark_id = marker.id // 10
        if marker.action == MARKER_DELETE:
            cache.pop(landmark_id, None)
            changed = True
            continue
        if marker.action != MARKER_ADD:
            continue

        position = transform(marker)
        if position is None:
            continue
        x, y = map(float, position)
        if math.isfinite(x) and math.isfinite(y):
            cache[landmark_id] = (x, y)
            changed = True
    return changed
