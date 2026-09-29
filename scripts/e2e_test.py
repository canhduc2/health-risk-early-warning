"""Kiểm thử end-to-end hệ thống qua HTTP API + trang tĩnh.

Cách chạy:
    python3 -m uvicorn src.api:app --port 8000   # terminal 1
    python3 scripts/e2e_test.py                  # terminal 2

Yêu cầu server đang chạy tại localhost:8000 với dữ liệu demo đã seed
(scripts/seed_demo_data.py). Bộ test tự dọn dẹp sau mình: bản ghi P999
và luật R_E2E_* đều bị xóa, chỉ để lại audit trail.
"""
import json, urllib.request

B = "http://localhost:8000"


def req(method, path, body=None, raw=False):
    r = urllib.request.Request(B + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r) as resp:
            data = resp.read()
            if raw:
                return resp.status, data
            try:
                return resp.status, json.loads(data)
            except Exception:
                return resp.status, {}
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  OK {name}")
    else:
        failed += 1
        print(f"  XX {name} — {detail}")


print("== 1. Trang & du lieu tinh ==")
s, b = req("GET", "/", raw=True)
check("GET / (app.html)", s == 200 and b"tab-rules" in b)
check("app.html co dark-mode + trend + CSV export",
      b"theme-btn" in b and b"trend-body" in b and b"exportCSV" in b)
s, b = req("GET", "/rules", raw=True)
check("GET /rules", s == 200)
s, b = req("GET", "/benchmark", raw=True)
check("GET /benchmark (co tab robust + cc panel)", s == 200 and b"tab-robust" in b and b"cc-body" in b)
s, d = req("GET", "/api/chat/patients")
check("/api/chat/patients co P001..P005 + DEMO",
      s == 200 and {"P001", "P005", "DEMO_HYPERTENSIVE"}.issubset(set(d.get("patients", []))), d.get("patients"))

print("== 2. Danh gia tung benh nhan seeded ==")
results = {}
for pid, expect_days, note in [("P001", 120, "lich su dai"), ("P003", 90, "trung binh"), ("P005", 5, "it ngay"),
                               ("DEMO_HYPERTENSIVE", 45, "HA tang"), ("DEMO_DIABETIC", 45, "duong tang")]:
    s, d = req("GET", f"/api/records/{pid}")
    check(f"{pid}: {len(d.get('dates',[]))} ngay ({note})", s == 200 and len(d.get("dates", [])) == expect_days)
    recs = [{"timestamp": t, "metric": m, "value": v["value"]} for t in d["dates"] for m, v in d["grid"][t].items()]
    s, a = req("POST", "/api/assess", {"patient_id": pid, "records": recs})
    ok = s == 200 and a.get("risk_level") in ("THAP", "TRUNG_BINH", "CAO", "INSUFFICIENT_DATA")
    ev = [e for e in a.get("evidence", []) if e.get("rule_id")]
    names_ok = all(e.get("rule") for e in ev)
    check(f"{pid}: level={a.get('risk_level')} score={a.get('risk_score')} luat={len(ev)} ten VN: {names_ok}", ok and names_ok)
    results[pid] = a

print("== 3. DEMO_HYPERTENSIVE: khong fire luat (9 luat = draft, chua approved) ==")
ev_ids = {e["rule_id"] for e in results["DEMO_HYPERTENSIVE"].get("evidence", []) if e.get("rule_id")}
check("Khong luat production nao kich hoat (C4: draft khong duoc dung dieu tri)", len(ev_ids) == 0, ev_ids)
print("     rules fire:", sorted(ev_ids))
md = results["DEMO_HYPERTENSIVE"].get("metrics_detail") or []
sb = [m for m in md if m["metric"] == "systolic_bp"]
if sb and sb[0].get("z_score") is not None:
    check(f"Tang 1 bat HA tam thu tang (z={sb[0]['z_score']:.2f})", sb[0]["z_score"] > 2)
else:
    check("Tang 1 z-score HA", False, "khong co z_score")

print("== 4. Records CRUD ==")
s, _ = req("PUT", "/api/records/P999", {"timestamp": "2026-08-23", "metric": "bmi", "value": 25.0})
check("PUT tao o moi", s == 200)
s, d = req("GET", "/api/records/P999")
check("GET thay o vua tao", d["grid"].get("2026-08-23", {}).get("bmi", {}).get("value") == 25.0)
s, _ = req("DELETE", "/api/records/P999?timestamp=2026-08-23&metric=bmi")
check("DELETE o", s == 200)
req("PUT", "/api/records/P999", {"timestamp": "2026-08-23", "metric": "spo2", "value": 97.0})
req("PUT", "/api/records/P999", {"timestamp": "2026-08-23", "metric": "heart_rate", "value": 70.0})
s, d = req("DELETE", "/api/records/P999?timestamp=2026-08-23")
check("DELETE ca ngay", s == 200 and d["deleted"]["metric"] == "*")

print("== 5. Governance ==")
s, kb = req("GET", "/api/kb")
check("GET /api/kb", s == 200 and len(kb.get("rules", [])) > 0)
body = {"rule_id": "R_E2E_TEST", "name": "Luat e2e", "system": "tim_manh",
        "condition": {"metric": "heart_rate", "op": ">", "threshold": 200},
        "severity": 0.5, "specialty": "Khoa Tim mach", "evidence": "guideline", "modes": ["all"], "actor": "tester"}
s, r = req("POST", "/api/kb/rules", body)
check("POST luat -> draft v1.0", s == 200 and r.get("rule", {}).get("status") == "draft" and r["rule"]["rule_version"] == 1.0)
rid = r["rule"]["rule_id"]
s, _ = req("POST", f"/api/kb/rules/{rid}/transition", {"to": "active", "actor": "tester"})
check("Chan nhay coc draft->active", s >= 400)
for to in ("review", "approved", "active"):
    s, r = req("POST", f"/api/kb/rules/{rid}/transition", {"to": to, "actor": "tester"})
check("review->approved->active + approved_by", s == 200 and r["rule"]["status"] == "active" and r["rule"]["approved_by"] == "tester")
s, r = req("PUT", f"/api/kb/rules/{rid}", {**body, "severity": 0.6})
v = r["rule"]["rule_version"]
st = r["rule"]["status"]
check("Sua luat active -> v1.1 draft", v == 1.1 and st == "draft", f"v{v}/{st}")
s, r2 = req("POST", "/api/assess", {"patient_id": "P001", "records": [{"timestamp": "2026-08-20", "metric": "heart_rate", "value": 210}]})
ev_ids2 = {e["rule_id"] for e in r2.get("evidence", []) if e.get("rule_id")}
check("Luat DRAFT khong cham production", rid not in ev_ids2, ev_ids2)
s, r = req("DELETE", f"/api/kb/rules/{rid}?actor=tester")
check("DELETE luat", s == 200)
s, d = req("GET", "/api/kb/audit?limit=100")
acts = [e["action"] for e in d["entries"]]
check("Audit du create/edit/delete/transition", all(a in acts for a in ("create", "edit", "delete", "transition")))

print("== 6. Markdown & evidence & research ==")
s, d = req("POST", "/api/render_markdown", {"markdown": "a\nb\nc"})
check("nl2br", "<br" in d.get("html", ""))
s, d = req("GET", "/api/evidence/ml?score=0.71")
check("evidence/ml raw+isotonic", "0.710" in d.get("html_md", "") and "isotonic" in d.get("html_md", ""))
s, d = req("GET", "/api/benchmark/research")
check("research du K2-K4+prov+status",
      all(k in d for k in ("label_sensitivity", "baseline_stability", "weight_sensitivity", "provenance", "evidence_status")))
states = [e["state"] for e in d["evidence_status"]]
check("Evidence status: done/partial/todo deu co", "done" in states and "todo" in states)
tv, cc = d.get("temporal_validation") or {}, d.get("complete_case") or {}
check("research co temporal_validation (EXP-TEMPORAL-LMF)",
      tv.get("lr", {}).get("auc_temporal") is not None, str(tv)[:120])
check("research co complete_case verdict", bool(cc.get("verdict")), str(cc)[:120])

print(f"\n=== KET QUA: {passed} PASS / {failed} FAIL ===")
raise SystemExit(1 if failed else 0)
