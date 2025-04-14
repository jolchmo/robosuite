
import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Tendon(ManipulatorModel):
    """
    Tendon Robot(6 wire 10 segment) robot model.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/tendon/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return None

    @property
    def default_gripper(self):
        return None

    @property
    def default_controller_config(self):
        return "default_tendon"

    @property
    def init_qpos(self):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @property
    def base_xpos_offset(self):
        return {
            # 按需修改
            "table": lambda table_length: (-table_length/3, 0, table_length+0.5)
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
