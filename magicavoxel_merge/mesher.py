from __future__ import annotations

import numpy as np


def greedy_mesh(voxels: np.ndarray, size: tuple[int, int, int], palette_rgba: np.ndarray) -> dict[str, np.ndarray]:
    sx, sy, sz = size
    if voxels.shape != (sx, sy, sz):
        raise ValueError("voxels shape does not match size")

    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    indices: list[int] = []

    def add_quad(p, du, dv, n, uv) -> None:
        base = len(positions)

        p0 = p
        p1 = (p[0] + du[0], p[1] + du[1], p[2] + du[2])
        p2 = (p1[0] + dv[0], p1[1] + dv[1], p1[2] + dv[2])
        p3 = (p[0] + dv[0], p[1] + dv[1], p[2] + dv[2])

        verts = (p0, p1, p2, p3)
        tri = (0, 1, 2, 0, 2, 3)

        v0 = np.asarray(verts[tri[0]], dtype=np.float32)
        v1 = np.asarray(verts[tri[1]], dtype=np.float32)
        v2 = np.asarray(verts[tri[2]], dtype=np.float32)
        fn = np.cross(v1 - v0, v2 - v0)
        if float(np.dot(fn, np.asarray(n, dtype=np.float32))) < 0.0:
            tri = (0, 2, 1, 0, 3, 2)

        positions.extend(verts)
        normals.extend((n, n, n, n))
        texcoords.extend((uv, uv, uv, uv))
        indices.extend([base + t for t in tri])

    dims = (sx, sy, sz)

    for d in range(3):
        u = (d + 1) % 3
        v = (d + 2) % 3

        x = [0, 0, 0]
        q = [0, 0, 0]
        q[d] = 1

        mask = np.zeros((dims[u], dims[v]), dtype=np.int32)

        # Sweep a cutting plane along axis d. Plane coordinate runs from 0..dims[d]
        # and compares voxels on each side: a at (x[d]-1) and b at x[d].
        for x[d] in range(dims[d] + 1):
            # Build face mask
            for x[v] in range(dims[v]):
                for x[u] in range(dims[u]):
                    a = 0
                    b = 0

                    ax = x[0] - q[0]
                    ay = x[1] - q[1]
                    az = x[2] - q[2]

                    bx = x[0]
                    by = x[1]
                    bz = x[2]

                    if 0 <= ax < dims[0] and 0 <= ay < dims[1] and 0 <= az < dims[2]:
                        a = int(voxels[ax, ay, az])
                    if 0 <= bx < dims[0] and 0 <= by < dims[1] and 0 <= bz < dims[2]:
                        b = int(voxels[bx, by, bz])

                    if (a != 0) == (b != 0):
                        mask[x[u], x[v]] = 0
                    elif a != 0:
                        # a is filled, b is empty => face points +d on plane x[d]
                        mask[x[u], x[v]] = a
                    else:
                        # a is empty, b is filled => face points -d on plane x[d]
                        mask[x[u], x[v]] = -b

            # Greedy merge over mask
            j = 0
            while j < dims[v]:
                i = 0
                while i < dims[u]:
                    c = int(mask[i, j])
                    if c == 0:
                        i += 1
                        continue

                    # Compute width
                    w = 1
                    while i + w < dims[u] and int(mask[i + w, j]) == c:
                        w += 1

                    # Compute height
                    h = 1
                    done = False
                    while j + h < dims[v] and not done:
                        for k in range(w):
                            if int(mask[i + k, j + h]) != c:
                                done = True
                                break
                        if not done:
                            h += 1

                    # Emit quad
                    x[u] = i
                    x[v] = j

                    normal = [0.0, 0.0, 0.0]
                    if c > 0:
                        normal[d] = 1.0
                        color_index = c
                    else:
                        normal[d] = -1.0
                        color_index = -c

                    pal_idx = int(color_index - 1)
                    uu = (float(pal_idx) + 0.5) / 256.0
                    vv = 0.5
                    uv = (uu, vv)

                    p = [float(x[0]), float(x[1]), float(x[2])]
                    p[d] = float(x[d])

                    du = [0.0, 0.0, 0.0]
                    dv = [0.0, 0.0, 0.0]
                    du[u] = float(w)
                    dv[v] = float(h)

                    add_quad(tuple(p), tuple(du), tuple(dv), tuple(normal), uv)

                    # Clear mask
                    for yy in range(h):
                        for xx in range(w):
                            mask[i + xx, j + yy] = 0

                    i += w
                j += 1

    return {
        "positions": np.asarray(positions, dtype=np.float32),
        "normals": np.asarray(normals, dtype=np.float32),
        "texcoords": np.asarray(texcoords, dtype=np.float32),
        "indices": np.asarray(indices, dtype=np.uint32),
    }


def greedy_quads(voxels: np.ndarray, size: tuple[int, int, int]) -> list[dict[str, object]]:
    sx, sy, sz = size
    if voxels.shape != (sx, sy, sz):
        raise ValueError("voxels shape does not match size")

    dims = (sx, sy, sz)
    out: list[dict[str, object]] = []

    for d in range(3):
        u = (d + 1) % 3
        v = (d + 2) % 3

        x = [0, 0, 0]
        q = [0, 0, 0]
        q[d] = 1

        mask = np.zeros((dims[u], dims[v]), dtype=np.int32)

        for x[d] in range(dims[d] + 1):
            for x[v] in range(dims[v]):
                for x[u] in range(dims[u]):
                    a = 0
                    b = 0

                    ax = x[0] - q[0]
                    ay = x[1] - q[1]
                    az = x[2] - q[2]

                    bx = x[0]
                    by = x[1]
                    bz = x[2]

                    if 0 <= ax < dims[0] and 0 <= ay < dims[1] and 0 <= az < dims[2]:
                        a = int(voxels[ax, ay, az])
                    if 0 <= bx < dims[0] and 0 <= by < dims[1] and 0 <= bz < dims[2]:
                        b = int(voxels[bx, by, bz])

                    if (a != 0) == (b != 0):
                        mask[x[u], x[v]] = 0
                    elif a != 0:
                        mask[x[u], x[v]] = a
                    else:
                        mask[x[u], x[v]] = -b

            j = 0
            while j < dims[v]:
                i = 0
                while i < dims[u]:
                    c = int(mask[i, j])
                    if c == 0:
                        i += 1
                        continue

                    w = 1
                    while i + w < dims[u] and int(mask[i + w, j]) == c:
                        w += 1

                    h = 1
                    done = False
                    while j + h < dims[v] and not done:
                        for k in range(w):
                            if int(mask[i + k, j + h]) != c:
                                done = True
                                break
                        if not done:
                            h += 1

                    x[u] = i
                    x[v] = j

                    normal = [0.0, 0.0, 0.0]
                    if c > 0:
                        normal[d] = 1.0
                        color_index = c
                    else:
                        normal[d] = -1.0
                        color_index = -c

                    p = [float(x[0]), float(x[1]), float(x[2])]
                    p[d] = float(x[d])

                    du = [0.0, 0.0, 0.0]
                    dv = [0.0, 0.0, 0.0]
                    du[u] = float(w)
                    dv[v] = float(h)

                    p0 = (p[0], p[1], p[2])
                    p1 = (p[0] + du[0], p[1] + du[1], p[2] + du[2])
                    p2 = (p1[0] + dv[0], p1[1] + dv[1], p1[2] + dv[2])
                    p3 = (p[0] + dv[0], p[1] + dv[1], p[2] + dv[2])

                    tri = (0, 1, 2, 0, 2, 3)
                    v0 = np.asarray([p0, p1, p2, p3][tri[0]], dtype=np.float32)
                    v1 = np.asarray([p0, p1, p2, p3][tri[1]], dtype=np.float32)
                    v2 = np.asarray([p0, p1, p2, p3][tri[2]], dtype=np.float32)
                    fn = np.cross(v1 - v0, v2 - v0)
                    if float(np.dot(fn, np.asarray(normal, dtype=np.float32))) < 0.0:
                        p0, p1, p2, p3 = p0, p3, p2, p1

                    out.append(
                        {
                            "verts": (p0, p1, p2, p3),
                            "normal": (float(normal[0]), float(normal[1]), float(normal[2])),
                            "color": int(color_index),
                            "w": int(w),
                            "h": int(h),
                        }
                    )

                    for yy in range(h):
                        for xx in range(w):
                            mask[i + xx, j + yy] = 0

                    i += w
                j += 1

    return out


def greedy_quads_baked(voxels: np.ndarray, size: tuple[int, int, int]) -> list[dict[str, object]]:
    sx, sy, sz = size
    if voxels.shape != (sx, sy, sz):
        raise ValueError("voxels shape does not match size")

    dims = (sx, sy, sz)
    out: list[dict[str, object]] = []

    for d in range(3):
        u = (d + 1) % 3
        v = (d + 2) % 3

        x = [0, 0, 0]
        q = [0, 0, 0]
        q[d] = 1

        # mask stores face presence and normal direction (+1 or -1)
        mask = np.zeros((dims[u], dims[v]), dtype=np.int8)
        # colors stores the color index for each face cell (0 when mask==0)
        colors = np.zeros((dims[u], dims[v]), dtype=np.int32)

        for x[d] in range(dims[d] + 1):
            for x[v] in range(dims[v]):
                for x[u] in range(dims[u]):
                    a = 0
                    b = 0

                    ax = x[0] - q[0]
                    ay = x[1] - q[1]
                    az = x[2] - q[2]

                    bx = x[0]
                    by = x[1]
                    bz = x[2]

                    if 0 <= ax < dims[0] and 0 <= ay < dims[1] and 0 <= az < dims[2]:
                        a = int(voxels[ax, ay, az])
                    if 0 <= bx < dims[0] and 0 <= by < dims[1] and 0 <= bz < dims[2]:
                        b = int(voxels[bx, by, bz])

                    if (a != 0) == (b != 0):
                        mask[x[u], x[v]] = 0
                        colors[x[u], x[v]] = 0
                    elif a != 0:
                        # a is filled, b is empty => face points +d on plane x[d]
                        mask[x[u], x[v]] = 1
                        colors[x[u], x[v]] = a
                    else:
                        # a is empty, b is filled => face points -d on plane x[d]
                        mask[x[u], x[v]] = -1
                        colors[x[u], x[v]] = b

            j = 0
            while j < dims[v]:
                i = 0
                while i < dims[u]:
                    c = int(mask[i, j])
                    if c == 0:
                        i += 1
                        continue

                    w = 1
                    while i + w < dims[u] and int(mask[i + w, j]) == c:
                        w += 1

                    h = 1
                    done = False
                    while j + h < dims[v] and not done:
                        for k in range(w):
                            if int(mask[i + k, j + h]) != c:
                                done = True
                                break
                        if not done:
                            h += 1

                    x[u] = i
                    x[v] = j

                    normal = [0.0, 0.0, 0.0]
                    normal[d] = float(c)

                    p = [float(x[0]), float(x[1]), float(x[2])]
                    p[d] = float(x[d])

                    du = [0.0, 0.0, 0.0]
                    dv = [0.0, 0.0, 0.0]
                    du[u] = float(w)
                    dv[v] = float(h)

                    p0 = (p[0], p[1], p[2])
                    p1 = (p[0] + du[0], p[1] + du[1], p[2] + du[2])
                    p2 = (p1[0] + dv[0], p1[1] + dv[1], p1[2] + dv[2])
                    p3 = (p[0] + dv[0], p[1] + dv[1], p[2] + dv[2])

                    tri = (0, 1, 2, 0, 2, 3)
                    v0 = np.asarray([p0, p1, p2, p3][tri[0]], dtype=np.float32)
                    v1 = np.asarray([p0, p1, p2, p3][tri[1]], dtype=np.float32)
                    v2 = np.asarray([p0, p1, p2, p3][tri[2]], dtype=np.float32)
                    fn = np.cross(v1 - v0, v2 - v0)
                    if float(np.dot(fn, np.asarray(normal, dtype=np.float32))) < 0.0:
                        p0, p1, p2, p3 = p0, p3, p2, p1

                    # Slice colors for this merged quad.
                    # colors is indexed by (u,v); convert to (v,u) for conventional image layout.
                    quad_colors = colors[i : i + w, j : j + h].T.copy()

                    out.append(
                        {
                            "verts": (p0, p1, p2, p3),
                            "normal": (float(normal[0]), float(normal[1]), float(normal[2])),
                            "color": 0,
                            "w": int(w),
                            "h": int(h),
                            "colors": quad_colors,
                        }
                    )

                    for yy in range(h):
                        for xx in range(w):
                            mask[i + xx, j + yy] = 0
                            colors[i + xx, j + yy] = 0

                    i += w
                j += 1

    return out
