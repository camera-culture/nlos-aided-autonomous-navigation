from cc_hardware.cnc_robot.gantry import GantryFactory
from cc_hardware.cnc_robot.controller import SnakeController
from cc_hardware.drivers.spads.tmf8828 import TMF8828Sensor
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.manager import Manager

# ==============

GANTRY_NAME: str = "SingleDrive1AxisGantry"
GANTRY_X_RANGE: tuple[float, float] = (0, 32)  # [cm]
GANTRY_Y_RANGE: tuple[float, float] = (0, 32)  # [cm]
GANTRY_NUM_X_STEPS: int = 4
GANTRY_NUM_Y_STEPS: int = 4

# ==============


def setup(manager: Manager, spad: TMF8828Sensor) -> None:

    gantry_controller = SnakeController(
        gantry=GantryFactory.create(GANTRY_NAME, port="/usr/local/dev/arduino-gantry"),
        x_range=GANTRY_X_RANGE,
        y_range=GANTRY_Y_RANGE,
        x_steps_per_direction=GANTRY_NUM_X_STEPS,
        y_steps_per_direction=GANTRY_NUM_Y_STEPS,
    )
    manager.add(gantry_controller=gantry_controller)


def capture(
    iter: int,
    *,
    manager: Manager,
    spad: TMF8828Sensor,
    gantry_controller: SnakeController,
) -> bool:
    get_logger().info(f"Capturing frame {iter}...")

    histogram = spad.accumulate(1, average=True)

    return gantry_controller.step(iter)


def main():
    get_logger().info("Starting capture...")

    with Manager(spad=TMF8828Sensor) as manager:
        manager.run(setup=setup, loop=capture)

    get_logger().info("Done.")


if __name__ == "__main__":
    main()
