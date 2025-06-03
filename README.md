# Tiny-Transformer Audio Anomaly Detector

Mô hình **Transformer Decoder** nhỏ gọn (kiểu GPT-style) phát hiện bất thường trong âm thanh
được biểu diễn dưới dạng chuỗi token rời rạc (từ K-Means clustering trên đặc trưng MFCC hoặc GFCC).

---

## 1  Pipeline Tổng Thể

```mermaid
flowchart LR
    A[Audio (.wav)] --> B[Trích xuất MFCC]
    B --> C[K-Means<br>→ token ID]
    C --> D[Phân loại dữ liệu]
    D -->|"normal"| E[Label: normal]
    D -->|"có nhãn lỗi"| F[Label: fault_x]
    D -->|"chưa gặp"| G[Label: undefined]
    E & F & G --> H[Tiny Transformer<br>(2 đầu ra)]
    H --> I[Anomaly Score]
    H --> J[Predicted Fault Class]
