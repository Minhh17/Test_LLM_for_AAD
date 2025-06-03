# Tiny-Transformer Audio Anomaly Detector

Mô hình **Transformer Decoder** nhỏ gọn (kiểu GPT-style) phát hiện bất thường trong âm thanh
được biểu diễn dưới dạng chuỗi token rời rạc (từ K-Means clustering trên đặc trưng MFCC hoặc GFCC).

---

## 1  Pipeline Tổng Thể
```mermaid
flowchart LR
    %% ---------- 1. Tiền xử lý ----------
    A["Audio (.wav)"] --> B["MFCC extraction"]
    B --> C["K-Means<br/>token ID"]

    %% ---------- 2. Gán nhãn ----------
    C --> D["Data labelling"]
    D -->|normal|     E["Label: normal"]
    D -->|fault_x|    F["Label: fault_x"]
    D -->|undefined|  G["Label: undefined"]

    %% ---------- 3. Huấn luyện ----------
    E & F & G --> H["Tiny Transformer<br/>2-head"]

    %% ---------- 4. Suy luận ----------
    H --> I["Anomaly score"]
    H --> J["Predicted fault class"]

