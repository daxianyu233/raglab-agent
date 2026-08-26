import json
from raglab.settings import PROJECT_ROOT
DATASET=PROJECT_ROOT/"raglab"/"evaluation"/"datasets"/"full_agent_e2e_v1.json"
def main():
    d=json.loads(DATASET.read_text(encoding="utf-8"))
    print("="*88); print("Phase 8D Full-Agent E2E Benchmark Design"); print("="*88)
    for c in d["cases"]:
        print(f"[{c['case_id']}] {c['category']}")
        print("  User：",c["user_input"])
        print("  Assertions：",", ".join(c["assertions"].keys()))
    print()
    print("当前仅完成 Benchmark / Ground Truth 设计；尚未绑定 SecureAgentRuntime。")
    print("后续 CLI / FastAPI / async / streaming 只需要实现同一 Adapter。")
if __name__=="__main__": main()
