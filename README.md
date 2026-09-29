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
./run_api.sh start        # khởi động API + hộp thoại chat (http://127.0.0.1:8000)
python -m src.main --input <file.csv> --output report/
```

## Giao diện & cách dùng ứng dụng

Sau khi chạy `./run_api.sh start`, mở trình duyệt tại **http://127.0.0.1:8000/chat**.

| Hộp thoại chat | Kết quả đánh giá |
|---|---|
| ![Giao diện chat](docs/screenshots/giao_dien_chat.png) | ![Kết quả đánh giá](docs/screenshots/ket_qua_danh_gia.png) |

**Cách dùng:**

1. Chọn hoặc nhập **mã bệnh nhân** ở góc trên (thanh xổ liệt kê các mã đã có dữ liệu trong `data/chat/`).
2. Nhập nhật ký sức khỏe hàng ngày — mỗi ngày một dòng, ví dụ:
   - `Huyết áp 135/85, nhịp tim 80`
   - `Đường huyết lúc đói 6.9, cân nặng 77`
   - Hoặc **đính kèm file PDF/DOCX** báo cáo khám — bằng nút 📎, **kéo-thả file**
     vào bất kỳ đâu trên trang, hoặc **dán (Ctrl+V)** file.
3. (Tùy chọn) Bật ô **"Ngày đo"** dưới khung nhập để **gán ngày cho lần ghi nhận**
   — hữu ích khi nhập lại nhật ký cũ hoặc nộp báo cáo khám nhiều ngày trước.
   Bỏ chọn → hệ thống dùng ngày hôm nay (hoặc ngày ghi trong tin nhắn).
4. Dữ liệu được tích lũy theo bệnh nhân; khi đủ 7 ngày đo, hệ thống đưa ra **báo cáo nguy cơ cá nhân hóa**.
5. Lệnh điều khiển: `trạng thái` (tình trạng hiện tại), `báo cáo` (đánh giá đầy đủ), `xóa dữ liệu` (reset bệnh nhân).

Kết quả đánh giá trình bày gọn: **mức rủi ro** (tiếng Việt: THẤP / TRUNG BÌNH /
CAO), **hệ cơ quan cần kiểm tra**, **cảnh báo theo luật lâm sàng** (mỗi luật kèm
link nguồn hướng dẫn gốc, số trang/chương/trích đoạn nếu có) và **chỉ số theo
dõi** (giá trị, đường cơ sở, thay đổi, xu hướng). Các chi tiết như công thức tính
điểm, chuẩn xếp loại, và **điểm tổng hợp theo từng mô hình ML** nằm trong thẻ
ẤN-MỞ (`<details>`) để báo cáo không rối — mỗi mô hình (XGB/LGBM/RF/FT/MLP/LR)
có một điểm cuối riêng (thống kê + tri thức y khoa + model đó + xu hướng), so
sánh được model nào đẩy mức rủi ro cao hơn. Báo cáo kết thúc bằng mức độ đầy đủ
dữ liệu (đánh dấu rõ chỉ số nào đang thiếu). Mọi suy luận ML đều ghi rõ là tham
khảo bổ sung, kết luận cuối do bác sĩ xác nhận — kết quả đủ thông tin để bệnh
nhân trình bác sĩ.

### Quản lý cơ sở tri thức (`/rules`)

Trang **http://127.0.0.1:8000/rules** dành cho bác sĩ: xem danh sách luật theo hệ cơ quan, **thêm/sửa/xóa luật** (nhập điều kiện chỉ số ghép AND/OR, độ nặng, chuyên khoa, link nguồn kèm số trang/chương/trích đoạn), **thêm hệ cơ quan mới** và **thêm chỉ số mới** — cơ sở tri thức mở rộng được mà không cần sửa code. Mọi thay đổi được validate trước khi ghi (sai cấu trúc, sai toán tử, độ nặng ngoài 0–1, link không hợp lệ đều bị từ chối kèm thông báo). Chi tiết: [docs/08](docs/08_Nguon_tri_thuc_va_xay_dung_luat.md).

API đầy đủ: **http://127.0.0.1:8000/docs** (Swagger).

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
**http://127.0.0.1:8000/benchmark** (gồm cả mức độ đầy đủ dữ liệu — lưu ý
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

docs/                         # báo cáo nghiên cứu theo giai đoạn (01 → 21)
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
