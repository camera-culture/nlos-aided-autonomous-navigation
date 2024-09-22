from cc_hardware.drivers.cameras.flir import GrasshopperFlirCamera
from cc_hardware.drivers.spads.tmf8828 import TMF8828Sensor

def main():
    camera = GrasshopperFlirCamera()
    spad = TMF8828Sensor()

if __name__ == "__main__":
    main()
