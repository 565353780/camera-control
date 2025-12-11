from uv_cube_gen.Module.xyz_mapper import XYZMapper

def demo():
    xyz_mapper = XYZMapper(
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    )

    points = [1, 2, 3]

    points_world = xyz_mapper.toWorld(points)

    points_local = xyz_mapper.toLocal(points_world)

    print(points)
    print(points_world)
    print(points_local)

    return True
