from pathlib import Path
from datetime import datetime

import cv2

from cc_hardware.cnc_robot.gantry import GantryFactory
from cc_hardware.cnc_robot.controller import SnakeController
from cc_hardware.drivers.cameras.flir import GrasshopperFlirCamera
from cc_hardware.drivers.spads.tmf8828 import TMF8828Sensor
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.manager import Manager
from cc_hardware.utils.writers import PklWriter
from cc_hardware.algos.aruco import ArucoLocalizationAlgorithm

# ==============

ARUCO_DICT: int = cv2.aruco.DICT_6X6_250
ARUCO_MARKER_SIZE: float = 8.25  # [cm]
ARUCO_ORIGIN_ID: int = 116
ARUCO_OBJECT_ID: int = 120
ARUCO_CAMERA_ID: int = 137

GANTRY_NAME: str = "SingleDrive1AxisGantry"
GANTRY_X_RANGE: tuple[float, float] = (0, 32)  # [cm]
GANTRY_Y_RANGE: tuple[float, float] = (0, 32)  # [cm]
GANTRY_NUM_X_STEPS: int = 10
GANTRY_NUM_Y_STEPS: int = 10

OUTPUT_PKL: Path = Path("logs") / datetime.now().strftime("%Y%m%d_%H%M%S") / "data.pkl"

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

    gantry_controller = SnakeController(
        gantry=GantryFactory.create(GANTRY_NAME, port="/usr/local/dev/arduino-gantry"),
        x_range=GANTRY_X_RANGE,
        y_range=GANTRY_Y_RANGE,
        x_steps_per_direction=GANTRY_NUM_X_STEPS,
        y_steps_per_direction=GANTRY_NUM_Y_STEPS,
    )
    manager.add(gantry_controller=gantry_controller)

    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    assert not OUTPUT_PKL.exists(), f"Output file {OUTPUT_PKL} already exists"
    manager.add(writer=PklWriter(OUTPUT_PKL))


def capture(
    iter: int,
    *,
    manager: Manager,
    camera: GrasshopperFlirCamera,
    spad: TMF8828Sensor,
    algo: ArucoLocalizationAlgorithm,
    gantry_controller: SnakeController,
    writer: PklWriter,
) -> bool:
    get_logger().info(f"Capturing frame {iter}...")

    poses, images = algo.run(show=True, return_images=True)
    histogram = spad.accumulate(1, average=True)

    writer.append(
        {
            "iter": iter,
            "poses": poses,
            "histogram": histogram,
            "images": images,
        }
    )

    return gantry_controller.step(iter)


def main():
    get_logger().info("Starting capture...")

    with Manager(camera=GrasshopperFlirCamera, spad=TMF8828Sensor) as manager:
        manager.run(setup=setup, loop=capture)

    get_logger().info("Done.")


if __name__ == "__main__":
    main()
