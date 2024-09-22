import cv2

from cc_hardware.drivers.cameras.flir import GrasshopperFlirCamera
from cc_hardware.drivers.spads.tmf8828 import TMF8828Sensor
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.manager import Manager
from cc_hardware.algos.aruco import ArucoLocalizationAlgorithm

# ==============

ARUCO_DICT: int = cv2.aruco.DICT_6X6_250
ARUCO_MARKER_SIZE: float = 8.25  # [cm]
ARUCO_ORIGIN_ID: int = 116
ARUCO_OBJECT_ID: int = 120
ARUCO_CAMERA_ID: int = 137

# ==============


def setup(manager: Manager, camera: GrasshopperFlirCamera, spad: TMF8828Sensor) -> None:
    algo = ArucoLocalizationAlgorithm(
        camera,
        aruco_dict=ARUCO_DICT,
        marker_size=ARUCO_MARKER_SIZE,
        origin_id=ARUCO_ORIGIN_ID,
        object_id=ARUCO_OBJECT_ID,
        camera_id=ARUCO_CAMERA_ID,
    )
    manager.add(algo=algo)


def capture(
    manager: Manager,
    camera: GrasshopperFlirCamera,
    spad: TMF8828Sensor,
    algo: ArucoLocalizationAlgorithm,
) -> None:
    algo.run(visualize=True)


def main():
    get_logger().info("Starting capture...")

    with Manager(camera=GrasshopperFlirCamera, spad=TMF8828Sensor) as manager:
        manager.run(setup=setup, loop=capture)

    get_logger().info("Done.")


if __name__ == "__main__":
    main()
