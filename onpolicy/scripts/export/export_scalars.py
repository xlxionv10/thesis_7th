#!/usr/bin/env python3
import argparse
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:
    print("matplotlib is required to export PNGs. Install with:")
    print("  /usr/local/bin/python3 -m pip install matplotlib")
    raise SystemExit(1)

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:
    print("tensorboard is required to read event files. Install with:")
    print("  /usr/local/bin/python3 -m pip install tensorboard")
    raise SystemExit(1)


def find_event_dirs(root):
    event_dirs = []
    for dirpath, _, filenames in os.walk(root):
        if any(f.startswith("events.out.tfevents") for f in filenames):
            event_dirs.append(dirpath)
    return event_dirs


def safe_name(tag):
    return tag.replace("/", "__")


def export_dir_for_event(root_out, root_log, event_dir):
    rel = os.path.relpath(event_dir, root_log)
    rel = "." if rel == "." else rel
    return os.path.join(root_out, rel)


def export_scalars(event_dir, out_dir, max_tags=None):
    ea = EventAccumulator(event_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if max_tags is not None:
        tags = tags[:max_tags]

    if not tags:
        return 0

    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for tag in tags:
        events = ea.Scalars(tag)
        if not events:
            continue
        steps = [e.step for e in events]
        values = [e.value for e in events]
        plt.figure(figsize=(6, 4))
        plt.plot(steps, values)
        plt.title(tag)
        plt.xlabel("step")
        plt.ylabel("value")
        plt.tight_layout()
        out_path = os.path.join(out_dir, safe_name(tag) + ".png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Export TensorBoard scalars to PNG")
    parser.add_argument("--logdir", required=True, help="TensorBoard log directory")
    parser.add_argument("--outdir", required=True, help="Output directory for PNGs")
    parser.add_argument("--max_tags", type=int, default=None, help="Limit number of tags per run")
    args = parser.parse_args()

    logdir = os.path.abspath(args.logdir)
    outdir = os.path.abspath(args.outdir)

    if not os.path.isdir(logdir):
        print(f"logdir not found: {logdir}")
        return 1

    event_dirs = find_event_dirs(logdir)
    if not event_dirs:
        print("No TensorBoard event files found under:")
        print(f"  {logdir}")
        return 1

    total = 0
    for event_dir in sorted(event_dirs):
        run_out = export_dir_for_event(outdir, logdir, event_dir)
        count = export_scalars(event_dir, run_out, max_tags=args.max_tags)
        total += count
        print(f"Exported {count} plots from {event_dir} -> {run_out}")

    print(f"Done. Total plots: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
