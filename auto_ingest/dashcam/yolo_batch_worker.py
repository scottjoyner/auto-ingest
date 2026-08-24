#!/usr/bin/env python3
"""yolo_batch_worker.py - GPU batch YOLO sidecar producer (schema-exact).

Emits CSVs matching the legacy contract consumed by auto_ingest.dashcam.
yolo_embeddings.parse_yolo_csv:
    Key,vehicle_id,confidence,classification,xywh,xyxy,Frame
    <stem>_<frame>_<cls>_<n>, <id>, <label>, "<conf>%", "[x,y,w,h]", "[x1,y1,x2,y2]", <frame>
Filename: <clipstem>_YOLOv8n.csv   (boxes in NATIVE source pixels)

usage:
  yolo_batch_worker.py --clip /path/2025_0921_133252_F.MP4 [--device 0]
                       [--fps 5] [--imgsz 1280] [--conf 0.25]
                       [--out-dir DIR] [--model yolov8n.pt]
"""
import argparse, os, subprocess, sys, tempfile, time
import csv

def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True)

def probe_fps_w_h(path):
    r = sh(f'ffprobe -v quiet -print_format json -show_streams -select_streams v:0 "{path}"')
    import json
    j = json.loads(r.stdout or "{}")
    st = (j.get("streams") or [{}])[0]
    rate = st.get("avg_frame_rate", "30/1")
    num, den = (rate.split("/") + ["1"])[:2]
    fps = float(num) / float(den or 1)
    return fps, int(st.get("width", 0)), int(st.get("height", 0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--fps", type=float, default=5.0, help="sample rate for detection")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--out-dir", default=None, help="default: alongside clip in ../yolo/")
    args = ap.parse_args()

    clip = os.path.abspath(args.clip)
    stem = os.path.splitext(os.path.basename(clip))[0]
    out_dir = args.out_dir or os.path.dirname(clip)   # legacy contract: CSV beside video
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{stem}_YOLOv8n.csv")
    if os.path.exists(csv_path):
        print(f"SKIP {stem} (sidecar exists)"); return

    src_fps, W, H = probe_fps_w_h(clip)
    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix="yframes") as td:
        # decode natively (no scale) so boxes land in source pixel space
        r = sh(f'ffmpeg -nostdin -loglevel error -i "{clip}" '
               f'-vf "fps={args.fps}" -frame_pts 1 -q:v 2 "{td}/%08d.jpg"')
        frames = sorted(os.listdir(td))
        if not frames:
            print(f"EMPTY {stem}"); return

        import torch
        from ultralytics import YOLO
        dev = int(args.device)
        torch.cuda.set_device(dev)
        model = YOLO(args.model).to(f"cuda:{dev}")

        rows = []
        infer_t = 0.0
        for idx, fn in enumerate(frames, start=1):
            fp = os.path.join(td, fn)
            t = time.time()
            res = model.predict(fp, imgsz=args.imgsz, conf=args.conf,
                                device=dev, workers=0, verbose=False)[0]
            infer_t += time.time() - t
            names = res.names
            for bi, box in enumerate(res.boxes, start=1):
                cls = names.get(int(box.cls), str(int(box.cls)))
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                key = f"{stem}_{idx}_{cls}_{bi}"
                rows.append([
                    key, bi, cls, f"{float(box.conf)*100:.2f}%",
                    f"[{cx-w/2:.0f}, {cy-h/2:.0f}, {w:.0f}, {h:.0f}]",
                    f"[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]",
                    idx,
                ])

    with open(csv_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["Key","vehicle_id","confidence","classification","xywh","xyxy","Frame"])
        wtr.writerows(rows)

    dt = time.time() - t0
    print(f"WROTE {csv_path} | {len(rows)} det / {len(frames)} frames "
          f"| total {dt:.1f}s | pure-infer {infer_t:.1f}s ({len(frames)/max(infer_t,1e-9):.1f} FPS)")

if __name__ == "__main__":
    main()
