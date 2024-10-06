from pathlib import Path

import imageio
from tqdm import tqdm

from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.writers import PklWriter

def main(pkl_path: Path, outdir: Path | None = None):
    assert pkl_path.exists(), f"Input file {pkl_path} does not exist"

    outdir = outdir or pkl_path.parent / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    data = PklWriter.load_all(pkl_path)

    images = []
    for entry in tqdm(data):
        get_logger().info(f"Processing entry {entry['iter']}")

        assert len(entry["images"]) == 1
        images.append(entry["images"][0])

    get_logger().info(f"Saving video to {outdir / 'video.mp4'}")
    imageio.mimsave(outdir / "video.mp4", images, fps=30)


if __name__ == "__main__":
    import typer

    typer.run(main)