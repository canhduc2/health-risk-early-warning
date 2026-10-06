# ĐỀ CƯƠNG CHI TIẾT & HƯỚNG DẪN NỘI DUNG BÁO CÁO NCKH
**ĐỀ TÀI:** Nghiên cứu và xây dựng hệ thống cảnh báo sớm nguy cơ sức khỏe cá nhân hóa dựa trên khung hỗ trợ quyết định lâm sàng tích hợp học máy  
**Giảng viên hướng dẫn:** ThS. Nhữ Văn Kiên  
**Nhóm sinh viên thực hiện:** 
* Nguyễn Đức Cảnh (Chủ nhiệm – Chủ trì kiến trúc & mã nguồn)
* Nguyễn Khắc Nam Khánh (Nghiên cứu & Viết báo cáo)
* Vũ Đình An (Nghiên cứu & Viết báo cáo)

---

## 📌 QUY TẮC CỐT LÕI (RED LINES – AE BẮT BUỘC NẮM RÕ TRƯỚC KHI VIẾT)
1. **Định vị đề tài:** Đề tài xây dựng **Khung hỗ trợ quyết định lâm sàng (CDSF) đa tầng**, **KHÔNG** đề xuất một thuật toán Machine Learning mới. Mô hình LightGBM chỉ là một khối thành phần có thể thay thế.
2. **Không lạm dụng thuật ngữ "Xác suất Bayes":** Cơ chế tổng hợp điểm rủi ro $R$ là **tổng tuyến tính có trọng số kết hợp sàn an toàn (safety floor)**, tuyệt đối không được ghi là "suy luận mạng Bayes" vì không có phân phối tiên nghiệm, hàm hợp lý và hậu nghiệm.
3. **Phân định rạch ròi 3 bài toán dữ liệu (Tuyệt đối không đánh đồng):**
   * **NHANES gộp (16.314 người):** Là bài toán **phân loại kiểu hình cắt ngang**, có hiện tượng vòng lặp đặc trưng–nhãn (feature-label loop). Không dùng để khẳng định khả năng dự báo tương lai.
   * **NHANES-LMF (9.821 người):** Là bài toán **dự báo tử vong mọi nguyên nhân 12 tháng** (có mốc khám MEC và khoảng dự báo rõ ràng).
   * **MIMIC-IV v3.1 (546.028 lượt nhập viện):** Là bài toán **kiểm thử độ ổn định kỹ thuật (stress test)** trên năm dịch chuyển (shifted year), **KHÔNG** phải ngoại kiểm lâm sàng hay kiểm định theo thời gian lịch.
4. **Trạng thái 9 luật y khoa:** 9 luật hiện ở trạng thái **Bản nháp (Draft)** phục vụ nguyên mẫu kỹ thuật, chưa được hội đồng y khoa phê duyệt chính thức (`approved_by`, `approved_at`).
5. **Hạn chế thứ tự thời gian ở Tầng 1:** Cửa sổ trượt 90 ngày hiện tại là phân tích hồi cứu/offline (chưa trượt lùi $t-1$ và có nội suy/điền lùi), trung thực nêu rõ đây là phiên bản nguyên mẫu.

---

# MỤC LỤC TỔNG THỂ & YÊU CẦU NỘI DUNG TỪNG MỤC

```
MỤC LỤC
DANH MỤC HÌNH ẢNH
DANH MỤC BẢNG BIỂU
DANH MỤC TỪ VIẾT TẮT
MỞ ĐẦU
1. Tính cấp thiết của đề tài
2. Mục tiêu nghiên cứu
3. Đối tượng và phạm vi nghiên cứu
4. Ý nghĩa khoa học và thực tiễn của đề tài
5. Cấu trúc của báo cáo

CHƯƠNG 1: TỔNG QUAN VỀ CƠ SỞ LÝ THUYẾT
1.1. Tổng quan về hệ thống hỗ trợ quyết định lâm sàng và xu hướng cá nhân hoá
1.2. Ứng dụng của trí tuệ nhân tạo và học máy trong cảnh báo y tế
1.3. Các khoảng trống nghiên cứu hiện tại
1.4. Đề xuất khung hỗ trợ quyết định lâm sàng (CDSF)
1.5. Tổng kết chương 1

CHƯƠNG 2: PHƯƠNG PHÁP NGHIÊN CỨU VÀ XÂY DỰNG HỆ THỐNG
2.1. Kiến trúc khung hỗ trợ quyết định lâm sàng (CDSF) đa tầng
2.2. Phương pháp tiền xử lý dữ liệu và trích chọn đặc trưng
2.3. Xây dựng Tầng 1: Phân tích thống kê và phát hiện bất thường cá nhân hóa
2.4. Xây dựng Tầng 2: Hệ chuyên gia và Cơ sở tri thức y khoa
2.5. Xây dựng Tầng 3: Mô hình học máy dự đoán rủi ro và Khả năng giải thích
2.6. Tích hợp hệ thống và giao diện ứng dụng
2.7. Tổng kết chương 2

CHƯƠNG 3: THỰC NGHIỆM VÀ KẾT QUẢ
3.1. Môi trường thực nghiệm và mô tả dữ liệu
3.2. Cài đặt và tích hợp cơ sở tri thức y khoa
3.3. Kết quả đánh giá mô hình học máy
3.4. Kết quả thực nghiệm hệ thống cảnh báo tích hợp qua các kịch bản
3.5. Thảo luận và đánh giá chung
3.6. Tổng kết chương 3

KẾT LUẬN VÀ KIẾN NGHỊ
TÀI LIỆU THAM KHẢO
PHỤ LỤC
```

---

## CHI TIẾT NỘI DUNG TỪNG MỤC (VIẾT GÌ - LẤY SỐ LIỆU GÌ - HÌNH NÀO)

---

### PHẦN MỞ ĐẦU
* **Mục tiêu:** Đặt vấn đề, nêu rõ lý do tại sao phải làm đề tài, mục tiêu và phạm vi.
* **Nội dung cần viết:**
  * **1. Tính cấp thiết:** EHR bùng nổ nhưng đối mặt với 3 thách thức: (1) Tính cá nhân hóa yếu (so sánh ngưỡng chung của cộng đồng bỏ qua đường cơ sở riêng); (2) Rào cản hộp đen của AI/Foundation Models; (3) Nhu cầu tích hợp tri thức hướng dẫn y khoa chuẩn mực.
  * **2. Mục tiêu nghiên cứu:** Mục tiêu tổng quát (xây dựng CDSF đa tầng hỗ trợ cảnh báo sớm và giải thích được) và 4 mục tiêu cụ thể (Tầng 1 thống kê cá nhân hóa, Tầng 2 luật tri thức JSON, Tầng 3 mô hình ML LightGBM + cơ chế tổng hợp điểm rủi ro, Tầng 4 XAI/SHAP và giao diện người dùng).
  * **3. Đối tượng & Phạm vi:**
    * Đối tượng: Dữ liệu chuỗi thời gian chỉ số sinh lý (huyết áp, nhịp tim, đường huyết, HbA1c, creatinine, eGFR, SpO2, BMI).
    * Phạm vi: Dữ liệu dạng bảng (tabular/time-series), không xử lý ảnh y tế. Hệ thống đóng vai trò **hỗ trợ quyết định lâm sàng (CDSS)**, không thay thế chẩn đoán bác sĩ.
  * **4. Ý nghĩa khoa học và thực tiễn:** Khung tích hợp kết nối thống kê cá nhân hóa với tri thức y khoa và ML, hỗ trợ truy vết quyết định.

---

### CHƯƠNG 1: TỔNG QUAN VỀ CƠ SỞ LÝ THUYẾT
* **Mục tiêu:** Khảo sát các công trình liên quan, chỉ rõ khoảng trống nghiên cứu để định vị đề tài.
* **Nội dung chi tiết từng mục:**
  * **1.1. Tổng quan về hệ thống hỗ trợ quyết định lâm sàng và xu hướng cá nhân hoá:**
    * Lịch sử phát triển từ CDSS dựa trên luật cổ điển sang khai thác dữ liệu dọc (Longitudinal EHR).
    * Tầm quan trọng của "đường cơ sở sinh lý cá nhân" (Personal Physiological Baseline): Cùng một chỉ số huyết áp 135 mmHg có thể là bình thường với người lớn tuổi nhưng là bất thường nghiêm trọng với người có baseline 105 mmHg.
  * **1.2. Ứng dụng của trí tuệ nhân tạo và học máy trong cảnh báo y tế:**
    * Các thuật toán truyền thống: Logistic Regression, Random Forest.
    * Các thuật toán Gradient Boosting hiện đại: XGBoost, LightGBM (rất mạnh trên dữ liệu dạng bảng có cấu trúc).
    * Các mô hình Deep Learning & Foundation Models gần đây: Transformer chuỗi thời gian, Foresight (Kraljevic et al., 2024), Delphi-2M (Shmatko et al., 2025).
  * **1.3. Các khoảng trống nghiên cứu hiện tại (Literature Gaps):**
    * *Đưa vào Bảng đối sánh:* **Bảng 1 trong bài báo** (Đối sánh 6 nghiên cứu tiêu biểu: Swinckels et al. [1], Foresight [4], Guo et al. 2023 [3], Guo et al. 2024 [2], Hänsel et al. [6], Feldman et al. [12]).
    * Phân tích 3 khoảng trống: (1) Rất ít mô hình có kiểm định ngoài thực tế (chỉ 2/20 nghiên cứu theo Swinckels); (2) Thiếu sự dung hòa giữa độ chính xác và tính minh bạch; (3) Chưa có khung nào tích hợp đồng bộ cả 3 nguồn: baseline cá nhân hóa + luật y khoa chuẩn + học máy.
  * **1.4. Đề xuất khung hỗ trợ quyết định lâm sàng (CDSF):**
    * Nêu ý tưởng cốt lõi: Thiết kế khung đa tầng tách biệt các luồng xử lý trước khi tổng hợp điểm. Khẳng định đây là giải pháp kiến trúc dung hòa giữa y học chứng cứ (evidence-based) và công nghệ dữ liệu.
  * **1.5. Tổng kết chương 1:** Tóm tắt ngắn gọn các luận điểm dẫn dắt sang Chương 2.

---

### CHƯƠNG 2: PHƯƠNG PHÁP NGHIÊN CỨU VÀ XÂY DỰNG HỆ THỐNG
* **Mục tiêu:** Trình bày chi tiết toán học, thuật toán, kiến trúc hệ thống và cách cài đặt kỹ thuật.
* **Nội dung chi tiết từng mục:**
  * **2.1. Kiến trúc khung hỗ trợ quyết định lâm sàng (CDSF) đa tầng:**
    * *2.1.1. Nguyên lý thiết kế hợp nhất đa nguồn bằng chứng:* Phân tách rõ ràng: Tín hiệu thống kê ($S$), Tri thức luật ($K$), Dự đoán học máy ($M$), Động lượng xu hướng ($T$).
    * *2.1.2. Sơ đồ kiến trúc tổng thể:* 
      * Vẽ sơ đồ luồng hệ thống: Cổng vào (CSV / PDF / Chat) $\rightarrow$ Lõi tiền xử lý $\rightarrow$ Tầng 1 (Detector) $\rightarrow$ Tầng 2 (Rule Engine) $\rightarrow$ Tầng 3 (RiskScorer) $\rightarrow$ Tầng 4 (XAI/SHAP) $\rightarrow$ Cổng ra (Báo cáo Markdown / REST API / Chat UI).
      * **Đưa Thuật toán 1 (Pseudocode của `assess_patient`)** vào mục này.
  * **2.2. Phương pháp tiền xử lý dữ liệu và trích chọn đặc trưng:**
    * *2.2.1. Tiền xử lý dữ liệu chuỗi thời gian:* Căn chỉnh ngày (`resample_to_daily`), lấy giá trị cuối cùng nếu có nhiều lần đo trong ngày. Xử lý giá trị khuyết (`impute_missing`): nội suy tuyến tính + ffill/bfill nếu khuyết $\le 30\%$; nếu $> 30\%$ thì giữ nguyên và đánh dấu cờ.
    * *2.2.2. Kỹ thuật trích chọn đặc trưng:* Trích xuất 7 đặc trưng nền (SBP, DBP, Heart Rate, Glucose, HbA1c, Creatinine, BMI) và các đặc trưng dẫn xuất (độ dốc, độ lệch Z-score, tỷ số EWMA).
    * *2.2.3. Sinh dữ liệu tổng hợp (Simulator) cho môi trường thiếu dữ liệu:* Trình bày công cụ tạo chuỗi giả lập có kiểm soát (control/shock) để kiểm thử biên hệ thống khi chưa có đầy đủ hồ sơ bệnh án thật.
  * **2.3. Xây dựng Tầng 1: Phân tích thống kê và phát hiện bất thường cá nhân hóa:**
    * *2.3.1. Thiết lập đường cơ sở sinh lý cá nhân:* Cửa sổ trượt 90 ngày, yêu cầu tối thiểu 5 điểm dữ liệu hợp lệ (`min_periods = 5`). Dưới 5 điểm trả về NaN, không phán đoán bừa.
    * *2.3.2. Định lượng độ lệch bằng Z-Score cá nhân hóa:*
      * Công thức Z-score: $Z_i(t) = \frac{x_i(t) - \mu_i(t)}{\sigma_i(t)}$.
      * Ngưỡng cảnh báo: Gắn cờ khi $|Z| \ge 2.0$.
    * *2.3.3. Đánh giá xu hướng bằng EWMA và STL:*
      * Công thức EWMA đệ quy: $S_t = \alpha x_t + (1-\alpha)S_{t-1}$ với $\alpha = 0.2$, ngưỡng biến thiên $\pm 2\%$.
      * Mô hình sai số dự báo 1 bước với $\alpha = 0.3$, ngưỡng $|z| \ge 2.5$.
      * STL Decomposition: Tách Trend, Seasonal, Residual để nhận diện suy giảm chức năng dài hạn (như eGFR giảm dần).
    * *2.3.4. Phát hiện dị thường đa chiều với Isolation Forest:* Cửa sổ 30 mốc gần nhất, `contamination = 0.05`, tối thiểu 10 mốc hợp lệ. Nhận diện các tương quan bất thường đa biến (ví dụ: huyết áp tăng kèm nhịp tim giảm).
    * *Đưa vào Bảng tham số thiết kế:* **Bảng 2 trong bài báo** (Tóm tắt toàn bộ tham số Tầng 1 và trọng số Tầng 3).
  * **2.4. Xây dựng Tầng 2: Hệ chuyên gia và Cơ sở tri thức y khoa:**
    * *2.4.1. Quy trình xây dựng cơ sở tri thức:* Cấu trúc JSON (`knowledge_base.json`). Vòng đời quản trị luật nghiêm ngặt: Draft $\rightarrow$ Review $\rightarrow$ Approved $\rightarrow$ Active. Nhật ký kiểm toán `audit_log.jsonl`.
    * *2.4.2. Xây dựng Rule Engine:* Bộ máy đánh giá điều kiện logic (AND, OR, lồng nhau) trên snapshot giá trị hiện tại của 10 chỉ số sinh lý.
    * *2.4.3. Ánh xạ chỉ số lâm sàng sang hệ cơ quan & Bảng 9 luật:*
      * Đưa **Bảng 3 trong bài báo** vào mục này: Bảng chi tiết 9 luật nguyên mẫu (R_CV_01, R_CV_02, R_CV_03, R_END_01, R_END_02, R_KID_01, R_KID_02, R_RES_01, R_MET_01) kèm nguồn gốc (ESC/ESH 2018, ADA 2023, KDIGO 2022, WHO) và **Ranh giới diễn giải** của từng luật.
  * **2.5. Xây dựng Tầng 3: Mô hình học máy dự đoán rủi ro và Khả năng giải thích:**
    * *2.5.1. Lựa chọn và thiết lập mô hình Học máy:* Kiến trúc LightGBM với 7 đặc trưng đầu vào, 300 cây, `learning_rate = 0.05`, `max_depth = 4`, `num_leaves = 16`.
    * *2.5.2. Thuật toán tổng hợp điểm rủi ro:*
      * Trình bày công thức tính 4 thành phần $S, K, M, T$:
        $$S = \min\left(1, \frac{\max_{r \in F} |z_r|}{4}\right), \quad K = \min\left(1, \max_{h \in H} \text{severity}_h\right), \quad M = \min(1, \text{ml\_score}), \quad T = \min\left(1, \frac{2 N_{rf}}{N_{\text{records}}}\right)$$
      * Công thức tổng hợp điểm thô:
        $$R_{\text{raw}} = 0.30S + 0.35K + 0.25M + 0.10T$$
      * Sàn an toàn lâm sàng (Safety floor): Nếu $\max(\text{severity}) \ge 0.7 \implies R = \max(R, 0.50)$.
      * Phân tầng rủi ro: THẤP ($R < 0.33$), TRUNG BÌNH ($0.33 \le R < 0.66$), CAO ($R \ge 0.66$).
    * *2.5.3. Minh bạch hóa quyết định với SHAP:* Giải thích đóng góp cục bộ của từng chỉ số vào điểm số của mô hình học máy.
  * **2.6. Tích hợp hệ thống và giao diện ứng dụng:**
    * *2.6.1. Xây dựng luồng trích xuất dữ liệu từ tài liệu y khoa:* 
      * Parser Regex dựa trên từ điển thuật ngữ song ngữ `patterns.json`.
      * Tùy chọn trích xuất ngữ nghĩa nâng cao qua LLM với cơ chế fallback tự động về Regex nếu mất kết nối.
    * *2.6.2. Phát triển API và giao diện người dùng:*
      * Hệ thống 21 RESTful API (`src/api.py`).
      * 3 phân hệ giao diện: Chat nhập liệu tương tác (`/chat`), Quản trị luật tri thức (`/rules`), và Bảng điều khiển kiểm chuẩn benchmark (`/benchmark`).

---

### CHƯƠNG 3: THỰC NGHIỆM VÀ KẾT QUẢ
* **Mục tiêu:** Trình bày toàn bộ số liệu thực nghiệm, bảng đối sánh, biểu đồ hiệu năng và kịch bản chạy thử. (Phần này trong docx đang trống, cần lấy toàn bộ dữ liệu từ bài báo và repo đưa vào).
* **Nội dung chi tiết từng mục:**
  * **3.1. Môi trường thực nghiệm và mô tả dữ liệu:**
    * *3.1.1. Môi trường và công cụ:* Python 3.10+, LightGBM, Scikit-learn, FastAPI, PyTorch, phần cứng huấn luyện.
    * *3.1.2. Mô tả các bộ dữ liệu y tế:* Đưa vào **Bảng 4 trong bài báo** (Mô tả 4 nhánh: Demo mô phỏng, NHANES gộp n=16.314, NHANES-LMF n=9.821, MIMIC-IV v3.1 n=546.028).
  * **3.2. Cài đặt và tích hợp cơ sở tri thức y khoa:**
    * *3.2.1. Cài đặt luật tim mạch & tăng huyết áp:* Chi tiết mã hóa logic R_CV_01 (HA tâm thu > 140 VÀ tâm trương > 90), R_CV_02 (nhịp tim > 100), R_CV_03.
    * *3.2.2. Cài đặt luật đái tháo đường & biến chứng thận:* Chi tiết R_END_01 (glucose đói > 7.0), R_END_02 (HbA1c > 6.5%), R_KID_01 (creatinine > 1.3), R_KID_02 (eGFR < 60).
  * **3.3. Kết quả đánh giá mô hình học máy:**
    * *3.3.1. Đối sánh kỹ thuật 6 mô hình trên NHANES gộp:*
      * Đưa vào **Bảng 5 trong bài báo** (ROC-AUC, PR-AUC, Brier score qua 5 seeds: XGBoost 0.9356, LightGBM 0.9349, Random Forest 0.9338, FT-Transformer 0.9257, MLP 0.8975, Logistic Regression 0.8844).
      * Đưa vào **Bảng 6 trong bài báo** (Phân tích độ nhạy complete-case vs suy diễn trung vị trên 52.05% dữ liệu thiếu glucose).
      * Đưa vào **Bảng 7 trong bài báo** (Hiệu chỉnh xác suất Platt vs Isotonic Regression, phân tích Brier và ECE).
    * *3.3.2. Đánh giá hiệu năng dự đoán rủi ro tử vong 12 tháng trên NHANES-LMF:*
      * Đưa vào **Bảng 8 trong bài báo** (Train 2015–2016, Test 2017–2018: Logistic Regression ROC-AUC 0.8209, LightGBM 0.7709, Harrell C-index 0.8217 và 0.7763).
      * Nhận xét: Mô hình tuyến tính đơn giản tỏ ra ổn định hơn trên dữ liệu nhiễu và tỷ lệ biến cố thấp (1.4%).
    * *3.3.3. Kiểm thử độ ổn định trên MIMIC-IV v3.1:*
      * Đưa vào **Bảng 9 trong bài báo** (Phân hoạch shifted year $\le 2115$ vs $\ge 2116$: ROC-AUC 0.7516 và 0.7508; Tỷ lệ bắt biến cố trong top 20% đạt 52.8% và 54.3%).
    * *3.3.4. Trực quan hóa và giải thích quyết định bằng XAI (SHAP):*
      * Biểu đồ SHAP Summary Plot (tầm quan trọng của các đặc trưng: Glucose đói, SBP, HbA1c dẫn đầu).
      * Biểu đồ SHAP Waterfall Plot trên một ca bệnh nhân cụ thể.
  * **3.4. Kết quả thực nghiệm hệ thống cảnh báo tích hợp qua các kịch bản (Case Studies):**
    * *3.4.1. Kịch bản 1: Ca khỏe mạnh (P001 - Lịch sử dài 120 ngày):*
      * Input: 120 ngày đo, 4 chỉ số HA, nhịp tim, BMI ổn định.
      * Output: Tầng 1 Z-score $\le +1.18\sigma$; Tầng 2 không có luật kích hoạt; Tầng 3 ML score 0.27 $\implies$ Điểm tổng hợp $R = 0.067$ (**THẤP**).
      * Ý nghĩa: ML có báo dương tính nhẹ nhưng nhờ Tầng 1 và 2 bình thường nên hệ thống không báo động giả.
    * *3.4.2. Kịch bản 2: Ca cảnh báo sớm nguy cơ đái tháo đường (DEMO_DIABETIC):*
      * Input: 45 ngày theo dõi, glucose và HbA1c tăng đột biến trong 7 ngày cuối.
      * Output: Tầng 1 bắt cờ dị thường Z-score; Tầng 2 kích hoạt R_END_01 và R_END_02 $\implies$ Điểm tổng hợp $R = 0.721$ (**CAO**).
    * *3.4.3. Kịch bản 3: Ca đánh giá nguy cơ tim mạch - tăng huyết áp (DEMO_HYPERTENSIVE):*
      * Input: 45 ngày theo dõi, huyết áp tâm thu tăng vọt lệch $+2.37\sigma$ so với chính baseline của bệnh nhân.
      * Output: Kích hoạt R_CV_01, R_CV_03, R_MET_01 $\implies$ Kích hoạt sàn an toàn 0.50 $\implies$ Điểm tổng hợp $R = 0.699$ (**CAO**).
  * **3.5. Thảo luận và đánh giá chung:**
    * *3.5.1. So sánh hiệu năng hệ thống CDSF đa tầng với các mô hình ML đơn lẻ:* Phân tích ưu thế của việc kết hợp đa nguồn so với việc chỉ tin vào một đầu ra duy nhất của ML.
    * *3.5.2. Đánh giá khả năng giải quyết rào cản "Hộp đen":* Nhờ báo cáo truy vết minh bạch (trích dẫn điều kiện luật, link y văn, Z-score cá nhân, giải thích SHAP), bác sĩ nắm rõ lý do đằng sau mỗi cảnh báo.
    * *3.5.3. Hạn chế của thực nghiệm:* Nêu trung thực 4 hạn chế (chưa thử nghiệm lâm sàng tiền cứu, 9 luật chưa qua hội đồng chuyên môn ký duyệt, rò rỉ thời gian nhẹ do nội suy/chưa shift-1 ở Tầng 1, MIMIC-IV là shifted-year).

---

### KẾT LUẬN VÀ KIẾN NGHỊ
* **Kết luận:** Tóm tắt 3 đóng góp chính của đề tài (Kiến trúc tích hợp 3 tầng, Cơ chế giải thích truy xuất nguồn gốc, Bộ thực nghiệm benchmark toàn diện trên NHANES và MIMIC-IV).
* **Kiến nghị & Hướng phát triển:** 
  * Hoàn thiện cơ chế cửa sổ trượt dịch lùi $t-1$ cho Tầng 1 để chạy online thời gian thực.
  * Mời hội đồng chuyên gia y khoa thẩm định và phê duyệt bộ luật vào trạng thái Active.
  * Tích hợp các mô hình chuỗi thời gian lớn (Chronos, TimesFM) vào nhánh sai số dự báo.
  * Triển khai thử nghiệm lâm sàng có kiểm soát tại cơ sở y tế.

---

### TÀI LIỆU THAM KHẢO (18 TÀI LIỆU CHUẨN APA)
Đưa toàn bộ danh mục 18 tài liệu tham khảo chuẩn mực từ bài báo khoa học vào báo cáo:
1. Swinckels, L., et al. (2024). JMIR, 26, e48320.
2. Guo, L. L., et al. (2024). npj Digital Medicine, 7, 171.
3. Guo, L. L., et al. (2023). Scientific Reports, 13, 3767.
4. Kraljevic, Z., et al. (2024). The Lancet Digital Health, 6(4), e281–e290.
5. Wornow, M., et al. (2023). npj Digital Medicine, 6, 135.
6. Hänsel, K., et al. (2023). Journal of Medical Systems, 47, 65.
7. CDC/NCHS. NHANES Survey.
8. CDC/NCHS. NHANES Linked Mortality Files.
9. PhysioNet. MIMIC-IV v3.1.
10. Shmatko, A., et al. (2025). Nature, 647, 248–256.
11. Shen, Y., et al. (2025). JMIR, 27, e59024.
12. Feldman, M. J., et al. (2025). JAMA Network Open, 8(5), e2512994.
13. Williams, B., et al. (2018). ESC/ESH Hypertension Guidelines. Eur Heart J.
14. Writing Committee Members, et al. (2023). ACC/AHA Guideline. Circulation.
15. American Diabetes Association (ADA). (2023). Standards of Care. Diabetes Care.
16. KDIGO. (2022). CKD Guidelines. Kidney Int.
17. WHO Consultation on Obesity. (2000). WHO TRS 894.
18. Harrell, F. E. (2015). Regression Modeling Strategies (2nd ed.). Springer.

---

## 👥 BẢNG PHÂN CÔNG CÔNG VIỆC

| Thành viên | Phụ trách chính | Nhiệm vụ cụ thể cần hoàn thành |
| :--- | :--- | :--- |
| **Nguyễn Đức Cảnh** *(Trưởng nhóm)* | Lõi kỹ thuật, Dữ liệu & Xuất biểu đồ | - Cung cấp toàn bộ hình ảnh biểu đồ thực nghiệm (ROC, PR, SHAP plot, biểu đồ nhịp tim/huyết áp Tầng 1).<br>- Chụp ảnh màn hình giao diện hệ thống (Trang Chat `/chat`, Quản trị luật `/rules`, Dashboard `/benchmark`).<br>- Rà soát kỹ thuật và duyệt toàn bộ báo cáo trước khi nộp. |
| **Nguyễn Khắc Nam Khánh** | Chương 1 & Chương 2 | - Viết hoàn thiện Phần Mở đầu và Chương 1 (Cơ sở lý thuyết, tổng quan, Bảng 1).<br>- Viết Chương 2 từ mục 2.1 đến 2.4 (Kiến trúc hệ thống, Thuật toán 1, Tiền xử lý, Tầng 1 Z-score/EWMA, Tầng 2 bộ luật JSON). |
| **Vũ Đình An** | Chương 2 & Chương 3, Kết luận | - Viết tiếp Chương 2 mục 2.5 và 2.6 (Tầng 3 scoring, Tầng 4 XAI, Pipeline ingest và API/UI).<br>- Viết toàn bộ Chương 3 (Thực nghiệm & Kết quả, điền đủ 5 bảng kết quả 5, 6, 7, 8, 9 và 3 kịch bản thực nghiệm).<br>- Viết Kết luận & Kiến nghị, chuẩn hóa 18 Tài liệu tham khảo. |