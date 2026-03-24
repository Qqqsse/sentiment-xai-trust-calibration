import os
import warnings
from typing import Dict, List, Tuple

import jieba
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as hf_logging


MODEL_NAME = "hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2"
OUTPUT_DIR = "outputs"
LIME_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "lime_result.png")
ATTN_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "attention_heatmap.png")


def configure_runtime_warnings() -> None:
    """
    避免已知、非致命的第三方套件雜訊警告干擾輸出。
    """
    warnings.filterwarnings(
        "ignore",
        message=".*clean_up_tokenization_spaces.*",
        category=FutureWarning,
    )
    # 避免 transformers 載入權重時輸出過多非致命訊息。
    hf_logging.set_verbosity_error()


def make_lime_output_path(tag: str) -> str:
    """
    為每個測試案例建立獨立 LIME 圖檔名，避免互相覆蓋。
    """
    safe_tag = (
        tag.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )
    return os.path.join(OUTPUT_DIR, f"lime_result_{safe_tag}.png")


def make_attention_output_path(tag: str) -> str:
    """
    為每個測試案例建立獨立 Attention 圖檔名，避免互相覆蓋。
    """
    safe_tag = (
        tag.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )
    return os.path.join(OUTPUT_DIR, f"attention_heatmap_{safe_tag}.png")


def ensure_chinese_font() -> None:
    """
    盡量設定常見中文字體；若找不到，印出提示訊息。
    """
    candidate_fonts = [
        "Microsoft JhengHei",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK TC",
        "Noto Sans CJK SC",
        "PingFang TC",
        "Heiti TC",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    selected = next((f for f in candidate_fonts if f in available), None)

    if selected:
        plt.rcParams["font.sans-serif"] = [selected]
        plt.rcParams["axes.unicode_minus"] = False
        print(f"[字體設定] 已使用中文字體: {selected}")
    else:
        warnings.warn(
            "找不到常見中文字體，圖表中文可能顯示為方塊。"
            "請安裝或手動指定字體，例如：\n"
            "plt.rcParams['font.sans-serif']=['Microsoft JhengHei']  # Windows 微軟正黑體\n"
            "plt.rcParams['font.sans-serif']=['SimHei']              # 黑體\n"
            "或使用 font_manager.fontManager.addfont('你的字體路徑.ttf') 載入後再設定。"
        )


def load_model_and_tokenizer(
    model_name: str = MODEL_NAME,
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        attn_implementation="eager",
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def infer_label_indices(model: AutoModelForSequenceClassification) -> Dict[str, int]:
    """
    從模型標籤中推測 Positive / Negative 的索引位置。
    若推測失敗，退回常見二分類預設: 0=Negative, 1=Positive。
    """
    id2label = getattr(model.config, "id2label", {}) or {}
    idx_neg, idx_pos = None, None
    for idx, label in id2label.items():
        text = str(label).lower()
        if "neg" in text:
            idx_neg = int(idx)
        if "pos" in text:
            idx_pos = int(idx)

    if idx_neg is None or idx_pos is None:
        idx_neg, idx_pos = 0, 1

    return {"Negative": idx_neg, "Positive": idx_pos}


def predict_proba(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
) -> np.ndarray:
    """
    回傳每筆文字的機率矩陣，shape=(N, num_labels)。
    """
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**encodings, output_attentions=True)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
    return probs


def explain_with_lime(
    sentence: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    save_path: str = LIME_OUTPUT_PATH,
) -> Tuple[str, Dict[str, float]]:
    """
    對單句做預測 + LIME 解釋，並儲存權重圖。
    """
    label_index = infer_label_indices(model)
    class_names = ["Negative", "Positive"]
    segmented_sentence = " ".join(jieba.lcut(sentence))

    def _lime_predict(batch_texts: List[str]) -> np.ndarray:
        # LIME 以空白分詞擾動；送進模型前把空白移除，回到自然中文句子。
        restored_texts = [text.replace(" ", "") for text in batch_texts]
        return predict_proba(restored_texts, tokenizer, model, device)

    probs = predict_proba([sentence], tokenizer, model, device)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = model.config.id2label.get(pred_idx, class_names[pred_idx])

    explainer = LimeTextExplainer(
        class_names=class_names,
        split_expression=r"\s+",
        bow=True,
    )
    exp = explainer.explain_instance(
        segmented_sentence,
        _lime_predict,
        labels=[label_index["Negative"], label_index["Positive"]],
        num_features=12,
        num_samples=1000,
    )

    target_label = label_index["Positive"] if pred_idx == label_index["Positive"] else label_index["Negative"]
    fig = exp.as_pyplot_figure(label=target_label)
    fig.set_size_inches(10, 5)
    fig.text(
        0.01,
        0.01,
        "顏色意義：綠色=提高目標類別機率；紅色=降低目標類別機率",
        fontsize=10,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    prob_info = {
        "Negative": float(probs[label_index["Negative"]]),
        "Positive": float(probs[label_index["Positive"]]),
    }
    neg_pos_sum = prob_info["Negative"] + prob_info["Positive"]
    if neg_pos_sum > 0:
        prob_info["Negative_rel"] = prob_info["Negative"] / neg_pos_sum
        prob_info["Positive_rel"] = prob_info["Positive"] / neg_pos_sum
    else:
        prob_info["Negative_rel"] = 0.0
        prob_info["Positive_rel"] = 0.0
    print(f"[LIME] 圖已儲存: {save_path}")
    return pred_label, prob_info


def plot_attention_heatmap(
    sentence: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    save_path: str = ATTN_OUTPUT_PATH,
    max_tokens: int = 30,
) -> None:
    """
    擷取最後一層 Attention，取 head 平均後畫熱力圖。
    """
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    if not outputs.attentions or len(outputs.attentions) == 0:
        raise RuntimeError(
            "模型未回傳 attention 權重，請確認模型支援 output_attentions=True，"
            "且已使用 attn_implementation='eager'。"
        )

    last_attn = outputs.attentions[-1][0]  # shape: [num_heads, seq_len, seq_len]
    mean_attn = last_attn.mean(dim=0).cpu().numpy()

    input_ids = inputs["input_ids"][0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # 移除特殊 token（如 [CLS], [SEP], [PAD]），聚焦於真實文字 token。
    valid_idx = [i for i, token_id in enumerate(input_ids) if token_id not in tokenizer.all_special_ids]
    if not valid_idx:
        raise RuntimeError("移除特殊 token 後沒有可視化內容，請檢查輸入句子。")

    valid_idx = valid_idx[:max_tokens]
    tokens = [tokens[i] for i in valid_idx]
    mean_attn = mean_attn[np.ix_(valid_idx, valid_idx)]

    # 以分位數設定色階，避免被極端注意力值主導。
    vmin = float(np.percentile(mean_attn, 5))
    vmax = float(np.percentile(mean_attn, 95))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        mean_attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="YlOrRd",
        square=True,
        cbar=True,
        vmin=vmin,
        vmax=vmax,
    )
    cbar = ax.collections[0].colorbar
    cbar.set_label("Attention 權重 (0~1)")
    plt.title("Last Layer Attention Heatmap (Mean over Heads)")
    plt.xlabel("被關注 token")
    plt.ylabel("發出關注 token")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.gcf().text(0.02, 0.01, "顏色意義：顏色越深(越紅)代表注意力權重越高", fontsize=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Attention] 熱力圖已儲存: {save_path}")


if __name__ == "__main__":
    configure_runtime_warnings()
    ensure_chinese_font()
    tokenizer, model, device = load_model_and_tokenizer(MODEL_NAME)

    samples = {
        "原句": "我已經等很久了，到現在還沒有回覆。",
        "測試A (改字)": "這部電影節奏稍慢，但也沒有到難看。",
        "測試B (刪字)": "這部電影節奏不快，但也沒有到。",
    }

    for tag, text in samples.items():
        print(f"\n===== {tag} =====")
        lime_path = make_lime_output_path(tag)
        attn_path = make_attention_output_path(tag)
        pred_label, probs = explain_with_lime(
            sentence=text,
            tokenizer=tokenizer,
            model=model,
            device=device,
            save_path=lime_path,
        )
        plot_attention_heatmap(
            sentence=text,
            tokenizer=tokenizer,
            model=model,
            device=device,
            save_path=attn_path,
        )
        print(f"句子: {text}")
        print(f"預測標籤: {pred_label}")
        print(f"Negative 機率 (全類別 Softmax): {probs['Negative']:.4f}")
        print(f"Positive 機率 (全類別 Softmax): {probs['Positive']:.4f}")
        print(f"Negative 相對機率 (僅 Neg/Pos): {probs['Negative_rel']:.4f}")
        print(f"Positive 相對機率 (僅 Neg/Pos): {probs['Positive_rel']:.4f}")
