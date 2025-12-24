from camera_control.Test.camera_pose import test as test_camera_pose
from camera_control.Test.uv import test as test_uv
from camera_control.Test.pnp import test as test_pnp
# from camera_control.Test.nvdiffrast_axis import test as test_nvdiffrast_axis

if __name__ == '__main__':
    test_camera_pose()
    # test_uv()
    test_pnp()
    # test_nvdiffrast_axis()
