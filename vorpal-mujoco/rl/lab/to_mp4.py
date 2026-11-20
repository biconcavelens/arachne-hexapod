"""GIF -> small H.264 MP4 next to it (GIFs are 10-70 MB, too big for git):  python lab/to_mp4.py a.gif"""
import subprocess
import sys

import imageio_ffmpeg

for gif in sys.argv[1:]:
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", gif, "-movflags", "+faststart",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "26", "-preset", "slow",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", gif[:-4] + ".mp4"], check=True)
