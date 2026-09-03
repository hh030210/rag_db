import json, os, sys
from pathlib import Path

results_dir = Path(__file__).parent / "Meta-Chunking" / "eval" / "CRUD" / "output"

# Two experiments: final (old chunks, top4) vs full_v2 (new chunks, top8)
experiments = {
    "final_top4": results_dir / "meta_chunks_final_top4_Qwen_API_Chat",
    "full_v2_top8": results_dir / "meta_chunks_full_v2_top8_Qwen_API_Chat",
}

task_names = {
    "QuestAnswer1Doc": "1-Doc QA",
    "QuestAnswer2Docs": "2-Docs QA",
    "QuestAnswer3Docs": "3-Docs QA",
}

for exp_name, exp_dir in experiments.items():
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*60}")
    for fname in sorted(exp_dir.glob("*.json")):
        task_key = fname.stem.replace("_Qwen_Qwen2.5-7B-Instruct", "")
        label = task_names.get(task_key, task_key)
        with open(fname, encoding="utf-8") as f:
            data = json.load(f)
        overall = data.get("overall", {})
        info = data.get("info", {})
        n_results = len(data.get("results", []))
        valid_results = [r for r in data.get("results", []) if r.get("valid", False)]
        print(f"\n  [{label}] {len(valid_results)}/{n_results} valid")
        if overall:
            for k, v in overall.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
        else:
            print(f"    (no overall scores)")
