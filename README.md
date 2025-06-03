# 🔊 Audio Anomaly Detection with a Tiny Transformer

> **Từ MFCC → Token K-means → Tiny GPT-style Transformer**  
> Phát hiện & phân loại lỗi âm thanh thời gian thực

## 1 · Tổng quan

1. **Rời rạc hoá audio**  
   - Trích xuất MFCC.  
   - Gom cụm K-means → chuỗi token cố định chiều dài.  

2. **Ngữ liệu**  
   | Loại   | Nhãn gốc | Ghi chú                              |
   |--------|----------|--------------------------------------|
   | Bình thường | `normal`   | Hoạt động tiêu chuẩn.           |
   | Bất thường đã biết | `fault_x` | Có mô tả cụ thể cho từng lỗi. |
   | Bất thường chưa biết | `undefined` | Sẽ được gán nhãn về sau.     |

3. **Mô hình** – Tiny Transformer decoder (vài block self-attention) với **2 đầu ra song song**  
   1. **Head #1** – *Language Modeling*  
      - Softmax → dự đoán **token tiếp theo**.  
      - Dùng **perplexity / NLL** làm **anomaly score**.  
   2. **Head #2** – *Fault Classification*  
      - Softmax → dự đoán **class lỗi** (`fault_x`).  
      - Chỉ huấn luyện trên mẫu **có nhãn lỗi**.

<p align="center">
 <img src="docs/pipeline.svg" width="650" alt="Pipeline overview">
</p>
