import argparse
from pathlib import Path
import re
import subprocess

from imageio_ffmpeg import get_ffmpeg_exe


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a DiscoveryWorld shared-map MP4.")
    parser.add_argument("frames_dir", type=Path, help="Run's frame directory (api.FRAME_DIR)")
    parser.add_argument("-o", "--output", type=Path, help="MP4 path (default: shared-world.mp4 in the run directory)")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second (default: 2)")
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("--fps must be at least 1")

    frame_numbers = sorted(
        int(match[1])
        for path in args.frames_dir.glob("shared_world_frame_*.png")
        if (match := re.fullmatch(r"shared_world_frame_(\d{4,})\.png", path.name))
    )
    if not frame_numbers:
        parser.error(f"No shared-world PNG frames found in {args.frames_dir}")
    if frame_numbers != list(range(frame_numbers[0], frame_numbers[-1] + 1)):
        parser.error("Shared-world frame numbering has gaps")

    output = args.output or args.frames_dir / "shared-world.mp4"
    if output.suffix.lower() != ".mp4":
        parser.error("The output path must end in .mp4")
    if output.exists():
        parser.error(f"Output already exists: {output}. Choose another path with --output.")

    subprocess.run(
        [
            get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n",
            "-framerate", str(args.fps), "-start_number", str(frame_numbers[0]),
            "-i", str(args.frames_dir / "shared_world_frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output),
        ],
        check=True,
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
