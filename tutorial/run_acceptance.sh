#!/usr/bin/env bash
# Acceptance test for the release: does the pipeline still work end to end?
#
# Run this before and after any change to src/. It is the difference between
# "the imports still resolve" and "the thing still trains and evaluates".
#
#   bash tutorial/run_acceptance.sh [outdir]
#
# ~6 minutes on one GPU. Isaac Sim does not exit when it is done (see
# tutorial/03_retarget/), so each stage is watched for its artefact and killed.
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD
OUT=${1:-$ROOT/logs/acceptance_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
DEMOS='["rt/0416_grasp/cube_small_1","rt/0416_grasp/cube_small_2","rt/0420_manip/squeegee_1","rt/0420_manip/squeegee_2"]'
EXTR=calib/camera_align/current.npy
fail=0

say () { echo; echo "=== $*"; }

# Fail fast and legibly rather than three stages deep. Activate your Isaac Lab
# conda env before running this; the script deliberately does not guess which
# one it is.
if ! python -c "import dexx, torch" 2>/dev/null; then
  echo "ERROR: `dexx` and `torch` must be importable."
  echo "       conda activate <your isaaclab env>  &&  pip install -e ."
  exit 1
fi

watch_for () {   # watch_for <artefact> <logfile> <timeout_units_of_10s> -- cmd...
  local art="$1" log="$2" lim="$3"; shift 4
  setsid nohup "$@" >>"$log" 2>&1 </dev/null & local pid=$!
  for _ in $(seq 1 "$lim"); do
    sleep 10
    [ -e "$art" ] && { sleep 15; pkill -9 -P "$pid" 2>/dev/null; kill -9 "$pid" 2>/dev/null; sleep 5; return 0; }
    kill -0 "$pid" 2>/dev/null || return 1
  done
  kill -9 "$pid" 2>/dev/null; return 2
}

say "1/4  static checks"
python tutorial/00_setup/check_install.py            > "$OUT/check_install.log" 2>&1 || fail=1
python tutorial/01_frames_and_constants/check_frames.py > "$OUT/check_frames.log" 2>&1 || fail=1
python tutorial/06_camera_calibration/inspect_extrinsic.py > "$OUT/inspect_extrinsic.log" 2>&1 || fail=1
python tutorial/00_setup/check_portable.py       > "$OUT/check_portable.log" 2>&1 || fail=1
python tutorial/00_setup/check_imports.py        > "$OUT/check_imports.log" 2>&1 || fail=1
tail -1 "$OUT/check_install.log"; tail -1 "$OUT/check_frames.log"
tail -1 "$OUT/check_portable.log"; tail -1 "$OUT/check_imports.log"

say "2/4  train a lean student (3 DAgger iterations)"
watch_for "$OUT/student/dagger_final.pth" "$OUT/train.log" 90 -- \
  python -u scripts/train_dagger_pc.py --task franka-sharpa-pointcloud \
    --teacher_ckpt checkpoints/teacher_poseobs.pth --side right --data_idx "$DEMOS" \
    --num_envs 64 --dagger_iters 3 --rollout_steps 2048 --train_epochs 2 \
    --batch_size 256 --hand_body_subset minimal6 \
    --student_drop_slots obj_bps,tips_distance,obj_pose_tail \
    --camera_extrinsic "$EXTR" --seed 0 --out_dir "$OUT/student" --headless \
  || { echo "TRAIN FAILED"; fail=1; }

say "3/4  evaluate it with physics DR on and a per-demo quota"
watch_for "$OUT/eval/records.json" "$OUT/eval.log" 90 -- \
  python -u scripts/eval.py --load_path "$OUT/student/dagger_final.pth" \
    --out_dir "$OUT/eval" --side right --data_idx "$DEMOS" \
    --num_envs 64 --max_episodes 40 --max_steps 4000 \
    --keep_physics_dr --camera_extrinsic "$EXTR" --headless \
  || { echo "EVAL FAILED"; fail=1; }

say "4/4  assert the artefacts"
python - "$OUT" <<'PY' || fail=1
import json, os, sys, torch
out = sys.argv[1]
ok = True
def chk(name, got, want):
    global ok
    good = got == want
    ok &= good
    print(f"  [{'ok' if good else 'FAIL'}] {name}: {got}" + ("" if good else f"  expected {want}"))

ck = torch.load(os.path.join(out, "student", "dagger_final.pth"),
                map_location="cpu", weights_only=False)
cfg = ck.get("cfg"); cfg = cfg if isinstance(cfg, dict) else vars(cfg)
chk("student proprio_dim", cfg.get("proprio_dim"), 417)
chk("hand keypoints", cfg.get("n_hand"), 6)
chk("scene points", cfg.get("n_scene"), 1024)
chk("tactile points", cfg.get("n_tactile"), 25)
chk("dropped slots", list(ck.get("student_drop_slots") or []),
    ["obj_bps", "tips_distance", "obj_pose_tail"])
chk("slot map", ck.get("student_obs_slots"),
    {"proprioception": (0, 79), "ref_tracking": (79, 390),
     "target_obj_pose": (390, 397), "tips_distance": (397, 402),
     "obj_bps": (402, 530), "tactile": (530, 550), "obj_pose_tail": (550, 557)})
w = [v for k, v in ck["model"].items() if k.endswith("mlp.0.weight") and v.shape[1] > 100][0]
chk("student MLP in_features", int(w.shape[1]), 481)

summ = json.load(open(os.path.join(out, "eval", "summary.json")))
# strict3 is the metric the protocol says to report; assert it is computed.
_strict = summ.get("strict") or {}
_has = all(k in _strict for k in ("strict2", "strict3", "strict5"))
ok &= _has
print(f"  [{'ok' if _has else 'FAIL'}] strict2/3/5 present in summary"
      + (f": strict3 = {100*_strict['strict3']['rate']:.1f}%" if _has else ""))
per = summ.get("success_rate_per_demo") or {}
counts = sorted({v["episodes"] for v in per.values()})
print(f"  [{'ok' if len(counts) == 1 else 'FAIL'}] episodes balanced across "
      f"{len(per)} demos: {counts}")
ok &= len(counts) == 1
print("\nACCEPTANCE: " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PY

echo
if [ "$fail" -eq 0 ]; then echo "ACCEPTANCE TEST PASSED   ($OUT)"; else echo "ACCEPTANCE TEST FAILED   ($OUT)"; fi
exit $fail
