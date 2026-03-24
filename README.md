# 文字情緒分析與 XAI／Trust Calibration 專案

本專案實作中文文字情緒分析流程，結合兩種可解釋性方法（LIME、Attention Heatmap）與兩種介面設計（Over-trust 與 Calibration），用來觀察模型解釋結果如何影響使用者信任。

## 專案重點

- 使用 Hugging Face 中文情緒模型進行情緒預測（含 attention 輸出）
- 以 `jieba` 解決中文 LIME 分詞問題，讓解釋粒度落在詞彙層級
- 以反事實測試（改字、刪字）驗證模型對關鍵詞依賴
- 輸出 LIME 權重圖與 Attention 熱力圖（每個測試句各自存檔）
- 提供兩個對照 UI：
  - `version_a_overtrust.html`：高權威、易導致盲信
  - `version_b_calibration.html`：中性校準、鼓勵人工覆核

## 專案結構

```text
sentiment_xai_project/
├─ src/
│  └─ analyze_model.py              # 情緒分析 + LIME + Attention + 反事實測試主程式
├─ ui/
│  ├─ version_a_overtrust.html      # 版本 A：Over-trust 介面
│  └─ version_b_calibration.html    # 版本 B：Trust Calibration 介面
├─ outputs/                         # 執行後產生圖檔（預設執行前可能為空）
├─ Report.md                        # 期末報告（Markdown）
├─ Report.pdf                       # 期末報告（PDF）
└─ requirements.txt                 # Python 依賴套件
```

## 環境需求

- Python 3.10+（建議）
- 網路連線（首次載入 Hugging Face 模型）

## 安裝方式

```bash
pip install -r requirements.txt
```

## 如何執行

```bash
python src/analyze_model.py
```

執行後會：

- 載入模型（`output_attentions=True`）
- 對三句測試語句進行預測與反事實分析
- 印出：
  - 預測標籤
  - `Negative/Positive` 全類別 Softmax 機率
  - `Negative/Positive` 相對機率（僅 Neg/Pos 重新正規化）
- 於 `outputs/` 產生每句對應圖檔：
  - `lime_result_<測試名稱>.png`
  - `attention_heatmap_<測試名稱>.png`

## 程式設計說明

### 1) 機率計算（Softmax）

- 在 `predict_proba()` 使用 `torch.nn.functional.softmax(logits, dim=-1)` 回傳正規化機率。
- 若模型含 `Neutral` 類別，`Negative/Positive` 的全類別機率可能偏小，屬正常現象；因此程式另提供僅 Neg/Pos 的相對機率，便於比較反事實變化。

### 2) LIME 中文分詞

- 先以 `jieba.lcut()` 將中文句子切詞，再以空白串接給 LIME。
- `LimeTextExplainer(split_expression=r"\s+")` 讓 LIME 以中文詞為最小單位解釋。
- 圖上已加註顏色意義：
  - 綠色：提高目標類別機率
  - 紅色：降低目標類別機率

### 3) Attention 熱力圖處理

- 擷取最後一層 attention，並對 head 取平均。
- 移除特殊 token（如 `[CLS]`、`[SEP]`、`[PAD]`），只保留真實文字 token。
- 使用分位數（5%~95%）設定 `vmin/vmax`，避免色階被極端值壓扁。
- 圖上含 colorbar 與顏色意義註記（顏色越深代表注意力越高）。

### 4) 多測試輸出不覆蓋

- LIME 與 Attention 皆採動態檔名（依測試標籤），每次測試獨立存圖，不會互相覆蓋。

## 測試樣本（反事實）

主程式預設使用以下三句：

1. 原句：`我已經等很久了，到現在還沒有回覆。`
2. 測試 A（改字）：`這部電影節奏稍慢，但也沒有到難看。`
3. 測試 B（刪字）：`這部電影節奏不快，但也沒有到。`

## UI 實驗頁面

- 開啟 `ui/version_a_overtrust.html`：觀察權威化設計如何提升盲信傾向。
- 開啟 `ui/version_b_calibration.html`：觀察不確定性揭露與反事實提問如何促進人工覆核。

## 報告文件

- `Report.md`：可直接閱讀完整分析內容
- `Report.pdf`：報告提交版本

## 常見問題

### Q1. 圖表中文字顯示方塊？

程式已做字體防呆，若系統無常見中文字體，請自行指定，例如：

```python
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]  # Windows 微軟正黑體
plt.rcParams["axes.unicode_minus"] = False
```

### Q2. 終端機出現 Hugging Face unauthenticated warning？

這不影響執行。若要提高下載速度與配額，可設定 `HF_TOKEN`。

### Q3. 為什麼 IDE 還顯示 import warning？

多半是編輯器選到不同 Python interpreter。請將 IDE interpreter 切換到已安裝 `requirements.txt` 套件的環境。

---

如需擴充，可進一步加入：

- 批次資料集評估與統計報表
- 不同模型的解釋一致性比較
- UI A/B 測試問卷與行為紀錄分析
