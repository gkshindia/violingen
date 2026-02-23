from violingen import Orchestrator

orch = Orchestrator(out_dir="output/separated", jobs=4)

stems = orch.process(["datasets/1.mp3"])
print("\nStem separation results:")
for src, out in stems.items():
    print(f"  {src} → {out}")

reports = orch.post_process(stems)  
print("\nPost-processing results:")
for r in reports:
    print(
        f"  {r['filename']}\n"
        f"    harmonic_ratio={r['harmonic_ratio']:.3f}  "
        f"contrast_ratio={r['contrast_ratio']:.2f}  "
        f"duration={r['duration']:.1f}s  "
        f"low_quality={r['low_quality']}"
    )
