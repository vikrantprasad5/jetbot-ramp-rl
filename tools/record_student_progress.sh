#!/bin/bash
# Record single-robot PiP clips of the camera student at several distillation stages and join them.
set -e
cd "$(dirname "$0")/.."
S=$(ls -d logs/rsl_rl/jetbot_ramp_student/*/ | tail -1)
FINAL=$(ls ${S}model_*.pt | sort -V | tail -1)
IT=$(basename "$FINAL" .pt | sed 's/model_//')
mkdir -p videos/phase3_parts
rec() {  # checkpoint, out, caption, steps, track
  ./run.sh tools/record_policy.py --task Isaac-JetbotRamp-Distill-Direct-Play --checkpoint "$1" --out "$2" \
    --caption "$3" --steps "$4" --track "$5" 2>&1 | grep "\[record\]"
}
rec ${S}model_0.pt  videos/phase3_parts/a_iter000.mp4 "Camera student - iteration 0 (untrained)" 240 0
rec ${S}model_30.pt videos/phase3_parts/b_iter030.mp4 "Camera student - iteration 30" 300 0
rec ${S}model_60.pt videos/phase3_parts/c_iter060.mp4 "Camera student - iteration 60" 300 0
rec ${S}model_80.pt videos/phase3_parts/d_iter080.mp4 "Camera student - iteration 80" 360 0
rec "$FINAL"        videos/phase3_parts/e_final.mp4   "Camera student - final (iteration $IT)" 450 0
ls videos/phase3_parts/*.mp4 | sed "s|^|file '$PWD/|; s|$|'|" > videos/phase3_parts/list.txt
ffmpeg -v error -y -f concat -safe 0 -i videos/phase3_parts/list.txt -c:v libx264 -pix_fmt yuv420p -crf 21 \
  -movflags +faststart videos/phase3_student_learning_progress.mp4
# final student on the mirrored S, 30 s (two laps)
rec "$FINAL" videos/phase3_student_final_mirrored.mp4 "Camera student - final, mirrored S, camera only" 900 1
echo done
