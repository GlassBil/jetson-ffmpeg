#!/usr/bin/env python3
"""Check enc_tear_soak output for torn frames.

Decode the harness's Annex-B stream to raw yuv420p first:

    ffmpeg -v error -i out.h264 -f rawvideo -pix_fmt yuv420p out.yuv

Then run:

    enc_tear_check.py out.yuv [width] [height]

Every source frame is a solid color from the 4-color cycle in enc_tear_soak.cpp, so each decoded
frame must be uniformly close to its expected color. Any region holding the previous frame's color
or zeroed chroma (the half-written-buffer signature) shows up as pixels far from the expected
value. Exit code 0 = clean, 1 = tears found.
"""

import sys

WIDTH = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
HEIGHT = int(sys.argv[3]) if len(sys.argv) > 3 else 800

# Must match COLORS in enc_tear_soak.cpp (Y, U, V).
COLORS = [(81, 90, 240), (145, 54, 34), (41, 240, 110), (210, 16, 146)]

TOLERANCE = 32       # per-pixel deviation allowed (codec noise on solid frames is far below this)
BAD_FRACTION = 0.001 # >0.1% deviant pixels in any plane = torn frame


def main():
    path = sys.argv[1]
    luma_size = WIDTH * HEIGHT
    chroma_size = (WIDTH // 2) * (HEIGHT // 2)
    frame_size = luma_size + 2 * chroma_size

    torn = []
    with open(path, "rb") as f:
        index = 0
        while True:
            data = f.read(frame_size)
            if len(data) < frame_size:
                break

            expected = COLORS[index % 4]
            planes = (
                ("Y", data[:luma_size], expected[0]),
                ("U", data[luma_size:luma_size + chroma_size], expected[1]),
                ("V", data[luma_size + chroma_size:], expected[2]),
            )

            worst = []
            for name, plane, value in planes:
                bad = sum(1 for p in plane if abs(p - value) > TOLERANCE)
                fraction = bad / len(plane)
                if fraction > BAD_FRACTION:
                    worst.append(f"{name}: {fraction:.1%} pixels off (expected {value})")

            if worst:
                torn.append((index, worst))

            index += 1

    print(f"{index} frames checked")
    if torn:
        for frame_index, details in torn:
            print(f"TORN frame {frame_index}: " + "; ".join(details))

        print(f"FAIL: {len(torn)}/{index} torn frames")
        return 1

    print("PASS: no torn frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
