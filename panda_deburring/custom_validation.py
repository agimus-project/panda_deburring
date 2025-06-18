import numpy as np
from rclpy.parameter import Parameter


def check_valid_quaternion(param: Parameter) -> str:
    """Checks if passed array od doubles can be used as a valid quaternion.

    Args:
        param (Parameter): ROS parameter with a double array containing quaternion.

    Returns:
        str: Error explanation. If empty string, everything is correct.
    """
    if not np.abs(np.linalg.norm(np.array(param.value)) - 1.0) < 1e-2:
        return (
            f"Invalid param '{param.name}' with value '{param.value}'. "
            "Norm of passed quaternion is not equal to '1.0'."
        )
    return ""
