#!/bin/bash
# Record single-robot PiP clips of the teacher at several training stages and join them into one video.
set -e
cd "$(dirname "$0")/.."
P=$(ls -d logs/rsl_rl/jetbot_ramp_teacher_progress/*/ | tail -1)
F=$(ls -d logs/rsl_rl/jetbot_ramp_teacher/*/ | tail -1)
FINAL=$(ls ${F}model_*.pt | sort -V | tail -1)
mkdir -p videos/phase2_parts
rec() {  # checkpoint, out, caption, steps, track
  ./run.sh tools/record_policy.py --task Isaac-JetbotRamp-Teacher-Direct-Play --checkpoint "$1" --out "$2" \
    --caption "$3" --steps "$4" --track "$5" 2>&1 | grep "\[record\]"
}
rec ${P}model_0.pt  videos/phase2_parts/a_iter000.mp4 "Teacher - iteration 0 (untrained, random weights)" 300 0
rec ${P}model_15.pt videos/phase2_parts/b_iter015.mp4 "Teacher - iteration 15 (about 6 s of training)" 300 0
rec ${P}model_25.pt videos/phase2_parts/c_iter025.mp4 "Teacher - iteration 25 (about 10 s of training)" 300 0
rec ${P}model_35.pt videos/phase2_parts/d_iter035.mp4 "Teacher - iteration 35 (about 14 s of training)" 360 0
rec "$FINAL"        videos/phase2_parts/e_final.mp4   "Teacher - final, 1500 iterations (about 9 min)" 450 1
ls videos/phase2_parts/*.mp4 | sed "s|^|file '$PWD/|; s|$|'|" > videos/phase2_parts/list.txt
ffmpeg -v error -y -f concat -safe 0 -i videos/phase2_parts/list.txt -c:v libx264 -pix_fmt yuv420p -crf 21 \
  -movflags +faststart videos/phase2_teacher_learning_progress.mp4
cp videos/phase2_parts/e_final.mp4 videos/phase2_teacher_final.mp4
echo done
