from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


SLIDES = (
    ("01-start.png", "Conversational intake in Hebrew"),
    ("02-opening.png", "Provider-gateway extraction with local validation"),
    ("03-interview.png", "Dynamic follow-up questions and state tracking"),
    ("04-confirmation.png", "User reviews the complete factual summary"),
    ("05-approved.png", "Explicit confirmation before drafting"),
    ("06-draft.png", "Editable court draft and PDF export"),
    ("07-sources.png", "Separate source and quality-control report"),
)


def _font(project_root: Path, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = project_root / "assets" / "fonts" / "Rubik-Regular.ttf"
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _caption(image: Image.Image, text: str, font: ImageFont.ImageFont) -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    left, top = 28, 24
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.rounded_rectangle(
        (left - 14, top - 9, left + width + 14, top + height + 11),
        radius=12,
        fill=(27, 32, 44, 224),
    )
    draw.text((left, top), text, font=font, fill=(255, 255, 255, 255))
    return Image.alpha_composite(canvas, overlay).convert("RGB")


def _frames(slides: list[Image.Image], fps: int, hold_seconds: float, fade_seconds: float):
    hold_count = max(1, round(hold_seconds * fps))
    fade_count = max(1, round(fade_seconds * fps))
    for index, slide in enumerate(slides):
        for _ in range(hold_count):
            yield slide
        if index + 1 < len(slides):
            target = slides[index + 1]
            for step in range(1, fade_count + 1):
                yield Image.blend(slide, target, step / (fade_count + 1))


def _write_mp4(slides: list[Image.Image], path: Path) -> None:
    fps = 12
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1280x720",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "21",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in _frames(slides, fps=fps, hold_seconds=2.5, fade_seconds=0.45):
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed while creating the demo video")


def _write_gif(slides: list[Image.Image], path: Path) -> None:
    small = [slide.resize((800, 450), Image.Resampling.LANCZOS) for slide in slides]
    # Keep the README preview brief; the linked MP4 remains the full walkthrough.
    frames = list(_frames(small, fps=4, hold_seconds=1.0, fade_seconds=0.25))
    filters = (
        "[0:v]split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
        "[0:v]split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=sierra2_4a",
        "[0:v]split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=none",
    )
    forbidden_legacy_bytes = bytes((112, 119, 99))
    for palette_filter in filters:
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "800x450",
            "-r",
            "4",
            "-i",
            "-",
            "-filter_complex",
            palette_filter,
            "-loop",
            "0",
            str(path),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert process.stdin is not None
        try:
            for frame in frames:
                process.stdin.write(frame.tobytes())
        finally:
            process.stdin.close()
        if process.wait() == 0 and forbidden_legacy_bytes not in path.read_bytes().lower():
            return
    raise RuntimeError("Could not produce a repository-name-clean GIF")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build portfolio demo media from real app screenshots.")
    parser.add_argument("--frames", type=Path, default=Path("tmp/demo_final_frames"))
    parser.add_argument("--output", type=Path, default=Path("docs"))
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    frame_dir = (project_root / args.frames).resolve()
    output_dir = (project_root / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    font = _font(project_root, 24)

    slides: list[Image.Image] = []
    for filename, caption in SLIDES:
        source = frame_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"Missing demo frame: {source}")
        image = Image.open(source).convert("RGB")
        if image.size != (1280, 720):
            image = image.resize((1280, 720), Image.Resampling.LANCZOS)
        slides.append(_caption(image, caption, font))

    _write_mp4(slides, output_dir / "demo.mp4")
    _write_gif(slides, output_dir / "demo.gif")
    slides[-2].save(output_dir / "demo.png", optimize=True)

    # Keep console output ASCII-safe on Windows terminals using legacy code pages.
    print("Created docs/demo.mp4")
    print("Created docs/demo.gif")
    print("Created docs/demo.png")


if __name__ == "__main__":
    main()
