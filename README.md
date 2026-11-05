# Nghiên cứu xây dựng khung đa tầng hỗ trợ đánh giá nguy cơ sức khỏe tích hợp phân tích cá nhân hóa, tri thức y khoa và học máy

*A Multi-Tier Framework for Personalized Health Risk Assessment Integrating Individual Analysis, Clinical Knowledge, and Machine Learning*

## Đội ngũ thực hiện

| Thành viên | Vai trò |
|------------|---------|
| Nguyễn Đức Cảnh | Trưởng nhóm — Chủ trì toàn bộ phát triển code |
| Nguyễn Khắc Nam Khánh | Nghiên cứu, tổng hợp và viết báo cáo |
| Vũ Đình An | Nghiên cứu, tổng hợp và viết báo cáo |

## Mục lục tài liệu

- [01. Tổng hợp nghiên cứu (5 bài báo)](docs/01_Tong_hop_nghien_cuu.md)
- [02. Kế hoạch dự án](docs/02_Ke_hoach_du_an.md)
- [03. Thiết kế hệ thống 3 tầng](docs/03_Thiet_ke_he_thong.md)
- [04. Lựa chọn & so sánh mô hình](docs/04_Lua_chon_mo_hinh.md)
- [05. Nâng cấp mô hình & Mô hình con trích xuất dữ liệu](docs/05_Mo_hinh_nang_cap_va_ingest.md)
- [06. Báo cáo huấn luyện: kết quả, giới hạn, hướng chuẩn mực hóa](docs/06_Bao_cao_huong_train_va_gioi_han.md)
- [07. Định vị đề tài: Khung hỗ trợ quyết định lâm sàng](docs/07_Dinh_vi_de_tai.md)
- [08. Nguồn tri thức y khoa và xây dựng luật](docs/08_Nguon_tri_thuc_va_xay_dung_luat.md)
- [09. Thảo luận & chuẩn hóa định vị đề tài (nội bộ)](docs/09_Thao_luan_chuan_hoa_dinh_vi.md)
- [10. Bài toán lựa chọn mô hình học máy](docs/10_Bai_toan_lua_chon_mo_hinh_ML.md)
- [11. Kết quả benchmark lần 1 (NHANES 1 chu kỳ)](docs/11_Ket_qua_benchmark_lan_1.md)
- [12. Báo cáo giai đoạn: trợ lý chat + đánh giá đa mô hình + benchmark mở rộng](docs/12_Bao_cao_giai_doan_chat_tong_hop.md)
- [13. Nhận xét của giảng viên hướng dẫn và đối sách của nhóm nghiên cứu](docs/13_Nhan_xet_giang_vien_va_doi_sach.md)
- [14. Mô tả chi tiết kiến trúc hệ thống: đầu vào/ra, dữ liệu huấn luyện, nhãn, hạn chế, kết quả](docs/14_Kien_truc_he_thong_chi_tiet.md)
- [15. Định hướng chuẩn hóa, kiểm chứng và hoàn thiện hệ thống (roadmap sau phản biện)](docs/15_Suy_nghi_va_lo_trinh_chuan_hoa.md)
- [16. Trả lời các vấn đề còn lại](docs/16_Tra_loi_cac_van_de_con_lai.md)
- [17. Kết quả sử dụng hệ thống mới](docs/17_Ket_qua_su_dung_he_thong_moi.md)
- [18. Giai đoạn P2: Dữ liệu dọc thời gian](docs/18_Giai_doan_P2_du_lieu_doc_thoi_gian.md)
- [19. Báo cáo tiến độ P2 — Đã giải quyết và giới hạn](docs/19_Bao_cao_tien_do_P2_da_giai_quyet_va_gioi_han.md)
- [20. Bộ khung bài báo AI4Industry 2026](docs/20_Bo_cuc_bai_bao_AI4Industry_2026.md)
- [21. Dàn ý bài báo AI4Industry 2026](docs/21_Dan_y_bai_bao_AI4Industry_2026.md)
- [28. Kế hoạch & trạng thái MCP Copilot Layer](docs/28_Ke_hoach_MCP_Copilot_Layer.md)

## Kiến trúc 3 tầng

```
Tầng 1  Phân tích bất thường & xu hướng cá nhân (Z-Score, Isolation Forest, EWMA/STL,
        sai số dự báo chuỗi thời gian — EWMA offline, sẵn sàng Chronos/TimesFM)
Tầng 2  Ánh xạ tri thức y khoa (rule engine trên JSON, 9 luật từ ESC/ESH, ADA, KDIGO, WHO)
Tầng 3  Tổng hợp rủi ro & hỗ trợ quyết định (6 mô hình ML XGB/LGBM/RF/FT/MLP/LR,
        tổng hợp có trọng số với trọng số là tham số thiết kế (design parameters) của nguyên mẫu)
```

## Kết quả chính

| Mô hình | ROC-AUC | Ghi chú |
|---|---|---|
| XGBoost | **0.9356±0.0028** | Đứng đầu benchmark |
| LightGBM | 0.9349±0.0031 | Model sản xuất |
| Random Forest | 0.9338±0.0038 | Gần tương đương |
| FT-Transformer | 0.9257±0.0043 | |
| MLP | 0.8975±0.0123 | |
| Logistic Regression | 0.8844±0.0078 | Baseline tuyến tính |

**Kiểm định temporally** (NHANES-LMF, train 2015-16 / test 2017-18):
- LR: AUC 0.821, Harrell C-index 0.822
- Lead time trung vị: 9 tháng (top quintile)

> **Định vị:** đề tài xây dựng một **khung hỗ trợ quyết định lâm sàng** (CDSF),
> không xây dựng một mô hình AI mới. Mô hình học máy (LightGBM) là một thành phần
> có thể thay thế trong khung. Chi tiết: [docs/07](docs/07_Dinh_vi_de_tai.md).

## Cài đặt

```bash
pip install -r requirements.txt
```

## Vận hành

```bash
./run_api.sh start        # Python AI core + API (http://127.0.0.1:8000)
./scripts/run_copilot.sh start   # Copilot bridge (http://127.0.0.1:8001)
python -m src.main --input <file.csv> --output report/
```

## Giao diện web mới (Vite + React + Node BFF)

Lớp web được tách thành SPA React và BFF Node, trong khi Python vẫn là lõi AI
duy nhất (không chuyển logic lâm sàng sang JavaScript):

```bash
cd web
cp .env.example .env       # cấu hình cổng, URL core/Copilot/Ollama
npm install
npm run dev                # Vite :5173, proxy /api sang BFF :3000
# hoặc production:
npm run build
npm start                  # BFF :3000, phục vụ dist/ và proxy /api
```

- Icon dùng `lucide-react`; không dùng emoji/character icon trong UI.
- Song ngữ EN/VI qua `react-i18next`; API Python nhận `lang=vi|en`.
- Copilot chỉ diễn đạt lại dữ kiện do tool cung cấp. Output LLM được kiểm tra
  tất định; nếu có chẩn đoán, dự đoán bệnh, lời khuyên thuốc, tên bệnh hoặc số
  liệu không có trong context thì **bỏ toàn bộ output LLM** và dùng câu trả lời
  tất định. Disclaimer cứng luôn được gắn bởi code.
- Trang `/copilot` hợp nhất **Copilot** và **Nhật ký sức khỏe** vào một luồng
  hội thoại, có nút chuyển **Tư vấn / Ghi nhật ký**. `/chat` tự chuyển hướng về
  `/copilot`; không còn hai trang chat trùng nhau.
- Câu hỏi kiểu “tình trạng hiện tại của bệnh nhân như nào?” hoặc
  “what is the current status?” được trả lời tất định từ hồ sơ đang chọn
  (`/api/chat/status` + `/api/records/{id}`), không bị coi là nhập chỉ số.
  Chỉ số còn thiếu trong form được điền từ lần đo mới nhất của hồ sơ.
- Dữ liệu demo: `.venv/bin/python scripts/seed_demo_patients.py`. Bộ seed không
  ghi đè hồ sơ đã có; P001 được làm giàu đủ 10 chỉ số trong 120 ngày để mọi
  biểu đồ và bảng đều có dữ liệu.
- BFF giữ nguyên đường dẫn Python: `/api/core/*` → `/api/*` và
  `/api/copilot/*` → `/api/copilot/*`. Khi sửa proxy, không bỏ phần rewrite này.

## Mô hình cục bộ (Ollama)

Cấu hình mặc định: **`qwen3:8b`** (Q4_K_M, ~5.2 GB, context 32.768 token),
lưu tại `D:\ollama\models` để tránh đầy ổ `C:`. Trên phần cứng hiện tại
(CPU + iGPU, không có NVIDIA GPU), `qwen3:8b` đạt khoảng 7.8 token/giây, phù
hợp cho phản hồi Copilot trong khoảng 20–30 giây. `qwen3:14b` cho chất lượng
cao hơn nhưng chỉ khoảng 4.1 token/giây, nên chỉ dùng khi ưu tiên chất lượng
hơn độ trễ. Trên Windows:

```powershell
setx OLLAMA_MODELS "D:\ollama\models"
setx OLLAMA_CONTEXT_LENGTH 32768
setx OLLAMA_KV_CACHE_TYPE q8_0
setx OLLAMA_NUM_PARALLEL 1
ollama pull qwen3:8b
```

Nếu chạy Python/Node trong WSL còn Ollama chạy trên Windows, đặt
`OLLAMA_URL` và `COPILOT_OLLAMA_URL` bằng IP gateway WSL
(ví dụ `http://172.30.16.1:11434`). Nếu cần chất lượng cao hơn và chấp nhận
độ trễ, có thể đổi sang `qwen3:14b` — cùng context 32k nhưng chậm hơn; đổi
`OLLAMA_MODEL`/`COPILOT_OLLAMA_MODEL` tương ứng.

## Giao diện & cách dùng ứng dụng

Sau khi chạy API lõi, Copilot bridge và BFF web, mở
**http://127.0.0.1:3000/copilot** (hoặc `/` cho trang đánh giá nguy cơ).

**Cách dùng:**

1. Chọn **mã bệnh nhân** ở thanh trên. Danh sách lấy từ dữ liệu đã có trong
   `data/chat/`; nút **Mới** tạo hồ sơ mới.
2. Trang **Đánh giá nguy cơ** (`/`) hiển thị bảng 10 chỉ số (phân trang 10 dòng),
   biểu đồ tương tác có tooltip, và kết quả đánh giá. Cột chỉ số vẫn hiển thị dù
   bệnh nhân chưa có giá trị.
3. Trang **Copilot & Nhật ký** (`/copilot`) dùng chung một luồng hội thoại:
   - **Tư vấn**: hỏi về kết quả, bằng chứng, mức nguy cơ, hoặc tình trạng hồ sơ
     bệnh nhân. Copilot trả lời từ dữ kiện tool và giữ đồng bộ với bệnh nhân
     đang chọn.
   - **Ghi nhật ký**: nhập chỉ số hàng ngày hoặc đính kèm PDF/DOCX/TXT. Có thể
     bật **Ngày đo** để gán ngày cho lần ghi; dữ liệu được lưu vào hồ sơ.
4. Sau khi đủ 7 ngày đo, bấm **Đánh giá & diễn giải** để xem mức nguy cơ, hệ cơ
   quan cần theo dõi, luật lâm sàng và bằng chứng.
5. Lệnh trong chế độ **Ghi nhật ký**: `trạng thái` (tình trạng hiện tại),
   `báo cáo` (đánh giá đầy đủ), `xóa dữ liệu` (reset bệnh nhân).

Kết quả đánh giá trình bày gọn: **mức rủi ro** (tiếng Việt: THẤP / TRUNG BÌNH /
CAO), **hệ cơ quan cần kiểm tra**, **cảnh báo theo luật lâm sàng** (mỗi luật kèm
link nguồn hướng dẫn gốc, số trang/chương/trích đoạn nếu có) và **chỉ số theo
dõi** (giá trị, đường cơ sở, thay đổi, xu hướng). Các chi tiết như công thức tính
điểm, chuẩn xếp loại, và **điểm tổng hợp theo từng mô hình ML** nằm trong thẻ
ẤN-MỞ (`<details>`) để báo cáo không rối. Mọi suy luận ML đều ghi rõ là tham
khảo bổ sung, kết luận cuối do bác sĩ xác nhận.

### Quản lý cơ sở tri thức (`/knowledge`)

Trang **http://127.0.0.1:3000/knowledge** dành cho bác sĩ: xem danh sách luật
theo hệ cơ quan, **thêm/sửa/xóa luật** (nhập điều kiện chỉ số ghép AND/OR, độ
nặng, chuyên khoa, link nguồn kèm số trang/chương/trích đoạn), **thêm hệ cơ quan
mới** và **thêm chỉ số mới** — cơ sở tri thức mở rộng được mà không cần sửa code.
Mọi thay đổi được validate trước khi ghi (sai cấu trúc, sai toán tử, độ nặng
ngoài 0–1, link không hợp lệ đều bị từ chối kèm thông báo). Chi tiết:
[docs/08](docs/08_Nguon_tri_thuc_va_xay_dung_luat.md).

API đầy đủ: **http://127.0.0.1:8000/docs** (Swagger).

## Copilot Layer (tùy chọn — lớp phủ độc lập)

**MCP Copilot** là lớp trợ lý hội thoại phủ lên hệ thống 3 tầng đã đóng băng:
hướng dẫn nhập đúng chỉ số/đơn vị, giải thích ý nghĩa từng tham số, gọi
`POST /api/assess` và diễn giải kết quả Tier 3 thành ngôn ngữ dễ hiểu. Trong
giao diện hợp nhất, chế độ **Tư vấn** dùng Copilot bridge (chỉ đọc hồ sơ qua
HTTP API); chế độ **Ghi nhật ký** gọi API lõi để ghi dữ liệu. Cả hai dùng chung
bệnh nhân đang chọn và cùng một luồng hội thoại. Disclaimer an toàn được hardcode
và luôn gắn bởi code, không do LLM sinh ra.

> **Lưu ý an toàn:** Kết quả chỉ là **hỗ trợ quyết định** cho bác sĩ,
> **không phải chẩn đoán**. Không dùng để tự mua, thay đổi hoặc ngừng thuốc.

Chạy (sau khi API cũ đã chạy ở cổng 8000):

```bash
pip install -r mcp_copilot/requirements.txt
./scripts/run_copilot.sh start          # mở http://127.0.0.1:8001
```

MCP server (tùy chọn, cần gói `mcp`):

```bash
python -m mcp_copilot                                   # stdio
python -m mcp_copilot --transport sse --port 8765       # SSE
```

Biến môi trường chính:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `COPILOT_CORE_API_URL` | `http://127.0.0.1:8000` | API lõi Python (ưu tiên; tránh trùng biến với BFF) |
| `COPILOT_API_URL` | `http://127.0.0.1:8000` | Fallback tương thích cho API lõi |
| `COPILOT_LLM_PROVIDER` | `auto` | `auto`/`ollama`/`openai`/`anthropic`/`none` |
| `COPILOT_ALLOW_CLOUD` | `0` | Phải bật rõ ràng mới cho phép gửi dữ kiện ra cloud |
| `COPILOT_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama cục bộ (WSL: dùng IP gateway Windows) |
| `COPILOT_OLLAMA_MODEL` | `qwen3:8b` | Model cục bộ, Q4_K_M |
| `COPILOT_OLLAMA_CONTEXT` | `32768` | Cửa sổ ngữ cảnh gửi kèm mỗi request |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | — | Chỉ dùng khi bật cloud |

Không có LLM → Copilot vẫn chạy 100% bằng câu trả lời tất định. Kiểm thử offline:

```bash
python3 scripts/test_copilot.py     # 66 test: parser, safety, luồng chat, small-talk, isolation
```

### Tối ưu hóa độ trễ LLM

- Model `qwen3:8b` chạy trên phần cứng CPU/iGPU (~7.8 token/giây). Để giảm độ trễ, Copilot dùng:
  - `keep_alive: 30m` để model luôn nạp sẵn (`COPILOT_OLLAMA_KEEP_ALIVE`).
  - Warmup khi server khởi động: nạp model và cache tiền tố prompt hệ thống (`/api/copilot/warmup`; frontend tự gọi khi vào trang `/copilot`).
  - LLM chỉ sinh **một câu tóm tắt** cho diễn giải kết quả (tối đa ~30 từ) và **tối đa 2 câu** cho câu hỏi tiếp theo, thay vì sinh toàn bộ phát hiện/khuyến nghị. Phát hiện và khuyến nghị vẫn được sinh ra tất định.
  - Context gửi LLM được gọn lại (`_compact_llm_context`): chỉ giữ mức nguy cơ, điểm số, hệ cần lưu ý, bằng chứng, dữ đủ, cờ khẩn cấp; bỏ các thành phần cồng kềnh.
  - Kết quả đoán: lần gọi explain đầu tiên ~14–20s (nếu model chưa nạp), các lần sau ~5–6s; follow-up ~5s; đánh giá tất định ~30ms.
- Nếu đầu ra LLM không qua kiểm duyệt an toàn (chẩn đoán, dự đoán bệnh, lời khuyên thuốc, số liệu không có trong context, injeciton), hệ thống **bỏ toàn bộ output LLM** và giữ nguyên câu trả lời tất định — không bao giờ làm sạch rồi chấp nhận.

### Chế độ gỡ lỗi LLM (chỉ để kiểm tra)

- Bật ô **"Debug LLM"** trên trang `/copilot` để xem raw output của Ollama, JSON đã parse, lý do kiểm duyệt, độ trễ, nhà cung cấp/mô hình, và kết quả chấp nhận/từ chối.
- Raw output chỉ dùng để kiểm tra (audit). Kết quả lâm sàng hiển thị luôn là tất định khi validation thất bại.
- Trường `method` trên mỗi trả lời assistants cho biết `llm_assisted` (dùng LLM) hay `deterministic` (không dùng LLM); mỗi tin nhắn trong hội thoại cũng gắn nhãn tương ứng cùng mô hình (ví dụ `LLM · qwen3:8b` vs `Tất định`).

### Hội thoại & giao diện

- Chào hỏi / cảm ơn / "bạn có thể làm gì" (`xin chào`, `hey`, `hi`, `what can you do`, …) được trả lời bằng Copilot (LLM-assisted khi bật, tất định khi tắt), không bị nhầm làm nhập chỉ số.
- Cuộn tự động xuống tin nhắn mới nhất.
- Mô hình hiện thị ở tiêu đề hội thoại (`Provider · Model`) và nhãn màu trên mỗi tin nhắn trợ lý (`badge` xanh da trời = LLM, xanh lá = tất định).
- Các thẻ phụ ("Giải thích chỉ số", "Trạng thái Copilot", "Debug LLM") được thu gọn mặc định để không làm loãng luồng đánh giá chính.

Chi tiết kiến trúc, quyết định và trạng thái: [docs/28](docs/28_Ke_hoach_MCP_Copilot_Layer.md).

## Huấn luyện mô hình

```bash
python scripts/build_nhanes_dataset.py   # dựng dataset thật NHANES (CDC)
python scripts/train_nhanes.py           # train model sản xuất trên NHANES
python scripts/download_datasets.py      # tải dataset chính thống (UCI)
python scripts/train_real_datasets.py    # đánh giá trên dữ liệu thật UCI
python scripts/run_benchmark.py          # benchmark 6 mô hình -> experiments/
```

## Benchmark

So sánh 6 mô hình học máy trên `data/datasets/nhanes_merged.csv` (NHANES 3 chu
kỳ, n = 16.314, 5 seed, impute median tránh data leakage). Kết quả tổng hợp ở
`experiments/summary.json` / `summary.md`, chi tiết từng seed ở
`experiments/EXP-ML-<MODEL>-<SEED>/`. Giao diện so sánh luận giải từng mô hình:
**http://127.0.0.1:3000/benchmark** (gồm cả mức độ đầy đủ dữ liệu — lưu ý
`glucose_fasting` thiếu 52% trong dataset gộp).

## Schema dữ liệu đầu vào

```csv
patient_id,timestamp,metric,value
P001,2025-01-01,systolic_bp,128
P001,2025-01-01,diastolic_bp,82
P001,2025-01-02,systolic_bp,131
```

## Cấu trúc mã nguồn

```
src/
  config.py                 # tham số toàn cục, ngưỡng, trọng số
  main.py                   # pipeline chính: nạp dữ liệu -> đánh giá -> báo cáo
  api.py                    # API FastAPI (chat, rules, benchmark, explain)
  core/                     # hạ tầng chung: pipeline, types, config
  data/                     # nạp / làm sạch / đặc trưng hóa dữ liệu (loader, features, preprocess)
  tier1_anomaly/            # phân tích bất thường & xu hướng cá nhân
  │                         # (zscore, isolation_forest, trend, forecast, detector)
  tier2_knowledge/          # cơ sở tri thức y khoa (JSON + rule engine)
  tier3_risk/               # điểm rủi ro (scoring) + báo cáo (report)
  tier4_explain/            # giải thích mô hình (SHAP + luận giải perturbation)
  ingest/                   # trích xuất dữ liệu từ PDF/DOCX/TXT (parsers, pipeline, extractor, llm_extractor)
  chat/                     # trợ lý trò chuyện: tích lũy nhật ký theo ngày, đánh giá, báo cáo
  │   ├─ agent.py           #   xử lý hội thoại, lệnh, đánh giá
  │   ├─ parser.py          #   nhận diện chỉ số từ câu tự nhiên
  │   ├─ store.py           #   lưu/truy vấn nhật ký theo bệnh nhân (JSONL)
  │   └─ static/            #   giao diện web: index.html (chat), rules.html, benchmark.html
  experiments/              # benchmark đa mô hình (models, runner, protocol, view, ft_transformer)

scripts/
  build_nhanes_dataset.py   # dựng dataset NHANES nhiều chu kỳ
  train_nhanes.py           # huấn luyện model sản xuất
  run_benchmark.py          # benchmark 6 mô hình x 5 seed -> experiments/
  download_datasets.py      # tải dataset UCI
  train_real_datasets.py    # đánh giá trên dữ liệu thật UCI
  export_nhanes_samples.py  # xuất báo cáo mẫu dạng PDF/DOCX (test ingest)
  gen_sample_data.py        # sinh dữ liệu mẫu
  fetch_nhanes_mortality.py # tải NHANES Linked Mortality Files
  run_temporal_validation.py # kiểm định temporally trên NHANES-LMF
  run_complete_case_check.py # kiểm định complete-case analysis
  e2e_test.py               # kiểm thử end-to-end (35 checks)
  run_copilot.sh            # start/stop/restart Copilot bridge (cổng 8001)
  test_copilot.py           # 62 test offline cho Copilot Layer

mcp_copilot/                # lớp Copilot độc lập (không import src/)
  tools.py                  #   5 công cụ: guide_input, explain_metric, call_assess,
  │                         #   explain_result, follow_up + handle_chat
  server.py                 #   MCP server stdio/SSE (tùy chọn)
  webapp.py                 #   FastAPI bridge cho giao diện chat (cổng 8001)
  static/copilot.html       #   giao diện chat legacy (không cần build)
  requirements.txt          #   phụ thuộc riêng (không đụng requirements.txt gốc)

web/                        # SPA React + BFF Node (giao diện chính)
  src/pages/CopilotPage.tsx #   Copilot & Nhật ký hợp nhất (Tư vấn / Ghi nhật ký)
  server/index.mjs          #   BFF proxy /api/core/* và /api/copilot/*

docs/                         # báo cáo nghiên cứu theo giai đoạn (01 → 28)
experiments/                  # evidence package + bảng tổng hợp benchmark
data/                         # dataset thật + dữ liệu chat mẫu (data/chat/, data/reports/)
```

## Tài liệu tham khảo chính

### Hướng dẫn lâm sàng (nguồn tri thức luật)

1. Williams, B., et al. (2018). 2018 ESC/ESH Guidelines for the management of arterial hypertension. *European Heart Journal*, 39(33), 3021–3104. https://doi.org/10.1093/eurheartj/ehy339
2. American Diabetes Association. (2023). Standards of Care in Diabetes — 2023. *Diabetes Care*, 46(Suppl 1), S1–S291. https://doi.org/10.2337/dc23-Sint
3. KDIGO 2022 Clinical Practice Guideline for Diabetes Management in CKD. *Kidney International*, 102(5S), S1–S127. https://doi.org/10.1016/j.kint.2022.06.008
4. World Health Organization. (2000). *Obesity: preventing and managing the global epidemic* (WHO TRS 894). https://iris.who.int/handle/10665/42330
5. Writing Committee Members et al. (2023). 2023 ACC/AHA/ACCP/HRS Guideline for the Diagnosis and Management of Atrial Fibrillation. https://doi.org/10.1161/CIR.0000000000001193
6. World Health Organization. (2019). *Guideline on use of pulse oximetry for monitoring of patients*. https://iris.who.int/handle/10665/345392

### Bộ dữ liệu

7. CDC/NCHS. (2020). National Health and Nutrition Examination Survey (NHANES) 2017–2018. https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017
8. CDC/NCHS. NHANES Linked Mortality Files 2019. https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/
9. UCI Machine Learning Repository. *Pima Indians Diabetes Database*. https://archive.ics.uci.edu/dataset/34/diabetes
10. UCI Machine Learning Repository. *Heart Disease (Cleveland) Data Set*. https://archive.ics.uci.edu/dataset/45/heart+disease

### Nghiên cứu liên quan (5 bài báo nền tảng)

11. Guo, L. L., Fries, J., Steinberg, E., et al. (2024). A multi-center study on the adaptability of a shared foundation model for electronic health records. *npj Digital Medicine*, 7(171). https://doi.org/10.1038/s41746-024-01166-w
12. Swinckels, L., Bennis, F. C., Ziesemer, K. A., et al. (2024). The Use of Deep Learning and Machine Learning on Longitudinal Electronic Health Records for the Early Detection and Prevention of Diseases: Scoping Review. *Journal of Medical Internet Research*, 26, e48320. https://doi.org/10.2196/48320
13. Kraljevic, Z., Bean, D., Shek, A., et al. (2024). Foresight — a generative pretrained transformer for modelling of patient timelines using electronic health records: a retrospective modelling study. *The Lancet Digital Health*, 6(4), e281–e290. https://doi.org/10.1016/S2589-7500(24)00025-6
14. Shmatko, A., Jung, A. W., Gaurav, K., et al. (2025). Learning the natural history of human disease with generative transformers. *Nature*, 647, 248–256. https://doi.org/10.1038/s41586-025-09529-3
15. Shen, Y., Yu, J., Zhou, J., & Hu, G. (2025). Twenty-Five Years of Evolution and Hurdles in Electronic Health Records and Interoperability in Medical Research: Comprehensive Review. *Journal of Medical Internet Research*, 27, e59024. https://doi.org/10.2196/59024

### Phương pháp

16. Harrell, F.E. (2015). *Regression Modeling Strategies*. Springer. https://doi.org/10.1007/978-3-319-19425-7

---

*Cập nhật lần cuối: 2026-09-03*
