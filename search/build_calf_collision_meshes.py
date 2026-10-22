"""Split the STL-derived calf hull at 2/3 of its longitudinal extent.

Run with the project Python. The two closed convex parts exactly cover the
previous hull; only the lower third is exempt from calf-contact penalties.
"""
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

ASSETS = (Path(__file__).resolve().parents[1] / "rl_x/environments/custom_mujoco/"
          "robot_locomotion/robots/silver_badger/data/assets")
UPPER_FRACTION = 2 / 3


def clipped_hull(vertices, faces, sign, boundary, lower):
    points = []
    for face in faces:
        triangle = vertices[face]
        for i, p in enumerate(triangle):
            q = triangle[(i + 1) % 3]
            dp, dq = sign * p[1] - boundary, sign * q[1] - boundary
            inside = dp >= 0 if lower else dp <= 0
            if inside:
                points.append(p)
            if dp * dq < 0:
                points.append(p + (q - p) * (-dp) / (dq - dp))
    points = np.unique(np.round(points, 10), axis=0)
    hull = ConvexHull(points)
    used = np.unique(hull.simplices)
    indices = {j: i + 1 for i, j in enumerate(used)}
    triangles = []
    for face, equation in zip(hull.simplices, hull.equations):
        a, b, c = points[face]
        if np.dot(np.cross(b - a, c - a), equation[:3]) < 0:
            face = face[[0, 2, 1]]
        triangles.append([indices[int(j)] for j in face])
    return points[used], triangles


def main():
    for side, sign in (("l", 1), ("r", -1)):
        source = ASSETS / f"l2{side}_calf_collision.obj"
        rows = [line.split() for line in source.read_text().splitlines()]
        vertices = np.array([[float(x) for x in row[1:]] for row in rows if row and row[0] == "v"])
        faces = np.array([[int(x) - 1 for x in row[1:]] for row in rows if row and row[0] == "f"])
        # Include the knee housing in the proximal part. The ratio is of the
        # actual collider's length, not the knee-to-foot joint spacing.
        longitudinal = sign * vertices[:, 1]
        boundary = longitudinal.min() + UPPER_FRACTION * np.ptp(longitudinal)
        for part in ("upper", "lower"):
            points, triangles = clipped_hull(vertices, faces, sign, boundary, part == "lower")
            output = ASSETS / f"l2{side}_calf_{part}_collision.obj"
            lines = [f"# {part} calf hull clipped from {source.name}; body frame, metres.",
                     f"# Proximal 2/3 + distal 1/3 of collider length; split |y|={boundary:.12f} m."]
            lines += ["v " + " ".join(f"{x:.10f}" for x in point) for point in points]
            lines += ["f " + " ".join(map(str, face)) for face in triangles]
            output.write_text("\n".join(lines) + "\n")
            print(output.name, len(points), "vertices", len(triangles), "faces")


if __name__ == "__main__":
    main()
