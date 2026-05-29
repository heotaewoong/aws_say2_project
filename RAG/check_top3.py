import json, glob

files = sorted(glob.glob("/home/ubuntu/diagnosis_report_*.json"))[-1:]
if not files:
    print("JSON 파일 없음")
else:
    print(f"파일: {files[0]}")
    with open(files[0]) as f:
        d = json.load(f)
    notes = d.get("clinical_notes", {})
    print("\n=== TOP1 ===")
    print(notes.get("top1_reasoning","")[:300])
    print("\n=== TOP2 ===")
    print(notes.get("top2_reasoning","")[:300])
    print("\n=== TOP3 ===")
    print(notes.get("top3_reasoning","")[:300])
    print("\n=== differential_note ===")
    print(notes.get("differential_note","")[:200])
    print(f"\n필드 키 목록: {list(notes.keys())}")
