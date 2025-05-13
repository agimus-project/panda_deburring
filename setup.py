from pathlib import Path
from typing import List

from generate_parameter_library_py.setup_helper import generate_parameter_module
from setuptools import find_packages, setup

package_name = "panda_deburring"
project_source_dir = Path(__file__).parent


def get_files(dir: Path, pattern: str) -> List[str]:
    return [x.as_posix() for x in (dir).glob(pattern) if x.is_file()]


setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/test_trajectroy_publisher"],
        ),
        ("share/panda_deburring", ["package.xml"]),
        (
            f"share/{package_name}/config",
            get_files(project_source_dir / "config", "*.yaml"),
        ),
        (
            f"share/{package_name}/launch",
            get_files(project_source_dir / "launch", "*.launch.py"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Guilhem Saurel",
    maintainer_email="guilhem.saurel@laas.fr",
    description="Deburring task implementation for Panda robot.",
    license="BSD",
    tests_require=["pytest"],
    entry_points={},
)
