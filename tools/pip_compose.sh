#!/bin/bash
# Picture-in-picture: put the robot-camera clip in the top-right quadrant of the viewer clip.
#
#   tools/pip_compose.sh <viewer.mp4> <robotcam.mp4> <out.mp4> ["caption text"]
#
# The robot camera (80x60, what the policy sees) is upscaled with nearest-neighbour to fit the
# 640x360 top-right quadrant of a 1280x720 viewer clip, keeping its 4:3 aspect, with a thin border.
set -e
VIEW=$1; CAM=$2; OUT=$3; CAPTION=${4:-}
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
[ -f "$FONT" ] || FONT=/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf
W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$VIEW")
H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$VIEW")
QH=$((H / 2)); QW=$((QH * 4 / 3))   # quadrant height, 4:3 inset width
X=$((W - QW - 12)); Y=12
LABEL="drawtext=fontfile=$FONT:text='robot camera':x=$((X + 8)):y=$((Y + QH - 28)):fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=4"
FILTER="[1:v]scale=${QW}:${QH}:flags=bicubic,pad=$((QW + 6)):$((QH + 6)):3:3:color=white[pip];[0:v][pip]overlay=$((X - 3)):$((Y - 3)):shortest=1,$LABEL"
if [ -n "$CAPTION" ]; then
  FILTER="$FILTER,drawtext=fontfile=$FONT:text='$CAPTION':x=16:y=h-40:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=6"
fi
ffmpeg -v error -y -i "$VIEW" -i "$CAM" -filter_complex "$FILTER" -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart "$OUT"
echo "$OUT"
