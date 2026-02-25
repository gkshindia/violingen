from violingen import Orchestrator

orch = Orchestrator(out_dir="output/separated", jobs=4)

stems = orch.process(["datasets/1.mp3"])
print("\nStem separation results:")
for src, out in stems.items():
    print(f"  {src} → {out}")
