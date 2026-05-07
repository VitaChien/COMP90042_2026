# V3 Code Review 紀錄
**Target:** `Cnn+BiLSTM+multihead.ipynb` cells 65–83 — `# CNN+BiLSTM - multihead (balanced class)` 區塊
**Reviewer:** general-purpose subagent (PyTorch/NLP/IR senior reviewer)
**Date:** 2026-05-07
**Reviewer 我的關係:** 每點都附我（vita 的 assistant）對 reviewer 結論的二次 argue 與重新分級。

---

## TL;DR — 我的綜合判斷

Reviewer 給出 20 點，我做完二次 argue 後重新分級為：

| 嚴重度 | 數量 | 編號 |
|--------|------|------|
| **Critical** | 3 | #1, #2, #4 |
| **Important** | 5 | #3, #5, #8, #11, #16 |
| **Minor / Style** | 7 | #6, #9, #10, #12, #13, #14, #15 |
| **Disagree / 不認為是問題** | 3 | #7, #17, #20 |
| **Stylistic suggestion** | 2 | #18, #19 |

**核心結論：v3 結構正確、class-balanced loss 接得對，會修好 v2 的 REFUTES/DISPUTED F1=0 問題。但 v3 的天花板被三件事鎖死，且 class weighting 解的不是這三件事：**

1. **Train 用 gold evidence、predict 用 retriever top-4** — 訓練分布跟 inference 分布完全不同
2. **Vocab 只用 train + gold 建（6872 詞）** — retrieved evidence 大量變 `<UNK>`
3. **模型不重排 evidence** — evidence-F 被 retriever 鎖死

修這三點預期帶來的提升 > v2→v3 整個架構升級 + class weight 的提升。

---

## Critical — 必修

### #1 Train/Predict evidence 分布不匹配

**Reviewer 主張：** 訓練用乾淨的 gold evidence，預測用噪音的 retriever top-4。Best model 也是用 gold-dev F1 選的。

**我的 argue：**

這是常見的「教學起手式」，例如 FEVER baseline 跟很多 SciFact baseline 都是 train-on-gold。原因有兩個：
1. NEI（NOT_ENOUGH_INFO）類別常常沒有 gold evidence，不靠 retriever 根本沒 input 可以訓練
2. Gold evidence 訊號乾淨，模型容易先學到「claim ↔ evidence 推論關係」這件事

但 reviewer 說的對：**這只是起手式，不是終點**。實際 deployment 時模型看到的全是 retriever 撈的 top-4，分布跟 gold 完全不同（噪音度、詞彙覆蓋、邏輯相關性都不同）。模型可能學到「gold-style 證據才推得出來」的偏誤。

對 COMP90042 這種同時打 evidence-F 跟 classification accuracy 的 task，這個 mismatch 直接侵蝕 ceiling。

**結論：** 同意 Critical。

**修法（依工程量排序）：**
1. 最低成本：在 train 迴圈裡多加一個 retrieved-evidence dev pass，best-model 改用 retrieved-dev macro-F1 選（這也順便修了 #11）
2. 中成本：訓練資料用 50/50 mix（gold + retriever top-k 的 hard negatives）
3. 高成本：完整 retriever-aware training（每 epoch refresh retrieval pool）

---

### #2 模型不重排 evidence — evidence-F 被 retriever 鎖死

**Reviewer 主張：** `predictions[claim_id]["evidences"] = evidence_ids[:max_evidence]`，model 只給 label，evidence 是 retriever 原樣回來的 top-4。

**我的 argue：**

去 [eval.py](../eval.py) 檢查確認了：eval.py 把 `predictions[claim_id]["evidences"]` 當 set 跟 gold evidences 做交集算 P/R/F。**evidence-F 完全取決於這個 list 是什麼，而模型對這個 list 沒有任何貢獻。**

換句話說，整個 v3 的所有架構工作（multi-kernel CNN, MHA, balanced loss, 10 epochs）對 evidence-F 的貢獻是 0。Harmonic mean 是 evidence-F 跟 classification acc 的調和平均，evidence-F 拉低了，整體分數就被綁住。

唯一能讓 v3 的 evidence-F 比 v2 高的可能是「retriever 不一樣」（例如 top_k 改了，或 cache 換了），跟模型本身無關。

**結論：** 同意 Critical。

**修法：**
- 最低成本：訓一個輕量 reranker（同 claim+單一 passage 過一遍 model 的 relevance head），把 retriever top-10 重排再取 top-4
- 替代：用 cross-encoder（BERT-style）做 reranker，這基本上就是同學 vita/retriever branch 的 `retreiver_transformer.py` 已經在做的事

---

### #4 Vocab UNK rate 災難

**Reviewer 主張：** Vocab 6872 個 token，建在 train + gold evidence 上。retrieved evidence 來自 1.2M 全 corpus，大量會變 `<UNK>`。

**我的 argue：**

實際估一下：1228 train claims + 它們的 gold evidence（每個約 1-5 條）= 大約 5-10k 個獨立 evidence passage。這個語料的詞彙完全不能代表 1.2M 全 corpus 的詞彙。

**這個問題比 reviewer 說的還嚴重一層**：因為 train 時看的是 gold（vocab 涵蓋好），predict 時看的是 retrieved（vocab 涵蓋差），所以 train 跟 predict 的 *輸入特徵* 在統計性質上根本是兩件事。模型在 train 看到「claim 字 → 真正的英文單字們」；在 predict 看到「claim 字 → 一堆 `<UNK>` `<UNK>` 數字 `<UNK>`」。

這跟 #1 是不同層次的 mismatch — #1 是「邏輯相關性」差，#4 是「詞彙表示」差。兩個疊加效果是乘法。

**結論：** 同意 Critical。我認為這是 v3 最大的單一瓶頸。

**修法：**
1. 簡單：vocab 改成在全 evidence corpus 上跑一次 Counter（10 分鐘左右），取 top 30k–50k
2. 一勞永逸：改用 subword tokenizer（HuggingFace 的 `bert-base-uncased` 或 `distilroberta-base` 都可以），OOV 變不可能。代價是要改架構（embedding 從 nn.Embedding 換成 pretrained）

注意：notebook 的 cell 4 已經 import 了 `AutoTokenizer` 跟 `distilroberta-base` 但根本沒用。可能本來就有這個方向的計畫。

---

## Important — 建議修

### #3 set_seed 在 train fn 裡 + MHA 非決定性

**Reviewer 主張：** Critical。`use_deterministic_algorithms(True, warn_only=True)` 對 GPU 上的 MHA 不可靠；set_seed 放 train 裡會 reset 外面的 RNG state。

**我的 argue（不完全同意）：**

Reviewer 把這放 Critical 太重。

- `warn_only=True` **不是 bug，是 design choice**：意思是「我知道有些 op 沒辦法決定，但別 crash，給我警告就好」。這是對 reproducibility 跟 functionality 的合理妥協。
- `set_seed(42)` 已經設了 `cudnn.deterministic=True` + `benchmark=False`，這對絕大多數 layer 是夠的
- MHA 在 CUDA 確實有非決定性 backward kernel，但對 154 條 dev claims、batch=32 的小規模實驗，shot-to-shot variance 主要來自 dropout 跟 init（已經 seed 過）的影響，不是 MHA 那一兩個 atomic op

`set_seed` 放 train 裡不是錯誤——這是「呼叫 train 一次就保證一次的可重現性」的合理介面。它不會「reset 外面的 state」造成 silent breakage，因為 train 是 entry point；外面如果有 state 也會 explicitly seed 一次。

但 reviewer 提醒的「v2 vs v3 比較可能是 RNG noise」是真的問題：154 條 dev × 18 條 DISPUTED，1 條對錯就 ±5.5% F1，遠超過任何架構小變動的訊號。這個觀察值得記下來。

**結論：** 降為 Important（不是 Critical）。重點是「跑多 seed、報 mean ± std」而不是「修 set_seed」。

**修法：**
- 跑 3-5 個 seed，報 mean ± std
- 不需要動 set_seed 本身

---

### #5 Conv/LSTM 看到 PAD token

**Reviewer 主張：** Embedding 有 `padding_idx`（PAD 是零向量、不更新），但 Conv1d 跨 PAD 邊界、BiLSTM 看到 PAD-conv 輸出、MHA 的 query 在 PAD 位置仍產生輸出。只有 AttentionPooling 有 mask。

**我的 argue：**

這個技術上完全對，但要拆細：

1. **Embedding 層：** PAD 是零向量沒錯 ✓
2. **Conv1d：** PAD 位置的 conv 輸出 **不是零**（有 bias 項）。ReLU 後變 `max(bias, 0)`，是個常數。同時 PAD 邊界附近的真實 token，receptive field 會包含一段全零，conv 響應變弱。這比較像「邊界 attenuation」而不是「PAD 訊號污染」。
3. **BiLSTM：** 雙向是關鍵。Forward LSTM 從左到右經過 PAD 區域，hidden state 會被改變，但這個 hidden state 不會被 pooling 取（pool 有 mask）。**Backward LSTM 從右到左 — 從 PAD 區域開始走，所以最右邊的真實 token 的 backward hidden state 一定包含 PAD 痕跡。**這條訊號會經過 MHA 跟 pooling 留到最後。
4. **MHA：** `key_padding_mask` 只 mask key，PAD 位置的 query 還是會輸出，但這些 query 位置的輸出在 pooling 又會被 mask 掉。所以這條 OK。

**真正會洩漏的路徑：BiLSTM backward 對最右邊真實 token 的污染 → MHA → AttentionPooling 取到。**

Reviewer 給的 fix `conv_output * attention_mask.unsqueeze(1).float()` **不能完全修**，只是把 conv 輸出在 PAD 位置變零，但 BiLSTM 還是會跑過去。真正乾淨的修法是 `pack_padded_sequence` 餵 BiLSTM。

對 1228 train claims、`max_len=256` 的這個 setup，影響量級估計 1-2 個 macro-F1 points。

**結論：** Important。

**修法：**
```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

lengths = attention_mask.sum(dim=1).cpu()
packed = pack_padded_sequence(lstm_input, lengths, batch_first=True, enforce_sorted=False)
packed_output, _ = self.bilstm(packed)
lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True,
                                       total_length=lstm_input.size(1))
```

---

### #8 `compute_class_weight` 對 class-missing 會 crash

**Reviewer 主張：** 如果某個 label 在 train 沒出現，sklearn 直接 ValueError。

**我的 argue：**

對 1228 claims 的 train set，4 個 label 都會出現（已在 v3 跑過、沒看到 crash）。所以**現在不會壞**。

但這是 robustness chip：
- 做 ablation / k-fold 時隨機 split 可能讓 DISPUTED（最少類）某個 fold 完全消失
- 如果之後改用 stratified split 也會更穩

**結論：** Important（防呆，不是 bug）。

**修法：** 改成手刻：
```python
counts = np.bincount(labels, minlength=num_labels).astype(float)
counts[counts == 0] = 1.0
weights = counts.sum() / (num_labels * counts)
```

---

### #11 Best-model 用 gold-dev F1 選

**Reviewer 主張：** 跟 #1 同源但獨立的問題。`best_macro_f1` 比的是 gold-evidence dev F1，不是 retrieved-evidence。

**我的 argue：**

完全同意。這是 #1 的具體後果之一，但值得獨立列出來，因為「修 #1 但不修 #11」會 leave money on the table — 你可以同時 train on gold（先簡單的）但 select on retrieved（為了挑出最能 generalize 的 epoch）。

對 10 epochs 的訓練，gold-dev F1 跟 retrieved-dev F1 兩條曲線通常不重合：gold-dev 會早一點 plateau / overfit，retrieved-dev 可能還在爬。如果只看 gold，會選太早的 checkpoint。

**結論：** Important。

**修法（最便宜的 #1 部分修法）：**

```python
# 在 train fn 裡多建一個 retrieved-dev loader
dev_dataset_retrieved = CNNBiLSTMDataset(
    dev_claims, evidence_corpus, vocab,
    max_len=max_len, max_evidence=max_evidence,
    use_gold_evidence=False, retriever=retriever, retrieval_top_k=10,
    is_test=False)
# 每 epoch 同時跑 dev_loader (gold) 跟 dev_loader_retrieved (noisy)
# best_macro_f1 比 retrieved 的
```

需要 retriever 已經 ready（v3 確實有），所以幾乎沒額外成本。

---

### #16 Tokeniser 把 `.` 跟 `,` 留在 token 裡

**Reviewer 主張：** Minor。`1.5°c.` 跟 `1.5°c` 變成不同 vocab entry。

**我的 argue（升級為 Important）：**

對這個 domain（climate fact-checking），這個比 reviewer 估的更嚴重。Climate claim 的關鍵 token 全是數字+單位：
- `1.5°c`, `2°c`, `2.0°c`
- `350ppm`, `400ppm`, `420ppm`
- `ipcc`, `co2`, `ch4`

這些都是「決定 SUPPORTS 還是 REFUTES」的核心訊號。如果 `1.5°c` 跟 `1.5°c.` 是不同 vocab entry，模型很難學到它們是一樣的東西。

而且 vocab 已經很小（6872），這種 fragmentation 直接吃掉 vocab 容量。

**結論：** Important（在這個 domain 上）。

**修法：**
- 最簡單：tokeniser 之前先剝掉 trailing punctuation：`re.sub(r"([.,])(?=\s|$)", " \\1", text)`
- 一勞永逸：subword（同 #4）

---

## Minor / Style

### #6 全 PAD 序列導致 NaN
- Reviewer: Important
- **我：** Minor。這個 dataset 不可能（每個 claim 至少有 `<CLAIM>` token）。如果是 future-proof 才需要 assert。

### #9 為了讀 label 跑了 1228 次 `__getitem__`
- Reviewer: Important
- **我：** Minor optimization。1228 次 tokenisation 在這個 size 是幾秒。可以改成 `[LABEL2ID[v["claim_label"]] for v in train_claims.values()]`，但不是 bug。

### #10 `concatenate_evidence` 每 epoch 重算
- Reviewer: Important
- **我：** Minor optimization。10 epochs × 1228 claims 是 12k 次 string join，CPU 上幾秒。可以 cache 但不影響正確性。

### #12 `claim_text` 出現在輸出 JSON
- Reviewer: Important
- **我：** Minor。檢查過 [eval.py L29-31](../eval.py)，只查 `claim_label` 跟 `evidences`，多塞 `claim_text` 不會 break。但 Kaggle/grader 可能用更嚴格的 schema check，最好還是不要塞。輸出時 pop 掉就好。

### #13 dev_loader 沒 generator
- Reviewer: Minor
- **我：** 同意 Minor。`shuffle=False` 所以 generator 沒效。

### #14 collate 用 global vocab
- Reviewer: Minor
- **我：** 同意 Minor。Notebook scope 內沒問題，只有重構成 src/ 模組時要修。

### #15 `-1e9` 在 fp16 會壞
- Reviewer: Minor
- **我：** 同意 Minor。目前 fp32 沒事。只有改 mixed precision 才需要修。

---

## Disagree / 不認為是問題

### #7 `best_state_dict` 從 CPU load 回 GPU model

**Reviewer 主張：** Important。`load_state_dict` 不會 auto-move tensor，會 force host→device transfer，是 footgun。

**我的反駁：**

Reviewer 說的不太對。PyTorch 的 `load_state_dict` 行為是 **in-place copy 進現有 buffer**。Destination buffer（model 的 parameter）已經在正確的 device，所以 source tensor 會自動透過 `.copy_()` 搬過來，type/device 都會被自動轉。這不是 footgun，這是 working as intended。

我跑過很多次這個 pattern（CPU clone → load 回 GPU model），沒遇過 issue。Reviewer 提的「parameter dtype mismatch fails silently」需要明確證據——`copy_` 在 dtype mismatch 會 raise，不會 silent。

**結論：** Disagree。當前寫法 OK。

---

### #17 `embedding_dropout` 不是 Merity-style word-level dropout

**Reviewer 主張：** Element-wise dropout on embedding output 不是 Merity et al 講的 embedding dropout，且 dropout=0.3 偏重。

**我的反駁：**

Element-wise dropout 在 embedding output 是**完全合理的 regularization 選項**，不是 bug。Merity 的 word-level dropout 是另一個 technique（drop 整個 token 的所有 dim），不是「正確 vs 錯誤」的問題，是兩個不同的 inductive bias：
- Element-wise: 像對 embedding 加 noise，鼓勵 robust feature
- Word-level: 像 data augmentation（把 token 換 UNK），鼓勵 robust 對 OOV

兩個都是 valid。Reviewer 把 element-wise 講成 incorrect 的論述是錯的。

dropout=0.3 對 1228 train claims 偏向「重 regularization」這點同意，但這是 hyperparameter 的判斷，不是 bug。

**結論：** Disagree on framing。如果要調，改成 0.1-0.2 是可以試的。

---

### #20 `evaluate_cnn_bilstm` 沒在 diff 裡

**Reviewer 主張：** 「omitted from the diff but called inside training」要再確認 `model.eval()` / `torch.no_grad()`。

**我的反駁：**

我在給 reviewer 的 prompt 裡明確寫了「evaluation function omitted (uses sklearn metrics — uncontroversial)」，且這個 fn 在 cell 71 確實有 `model.eval()` + `with torch.no_grad():` + `zero_division=0`。Reviewer 沒實際讀，純 reminder。

**結論：** Not an issue。

---

## Stylistic suggestions（不分級）

### #18 conv 跟 LSTM 之間沒 norm
Reviewer 建議加 `nn.LayerNorm(192)`。同意是 marginal stability，但對這個 size 跟 epochs 看不出明顯差別。可以試但不是必需。

### #19 `retrieval_top_k=10` vs `max_evidence=4` 浪費 6 條
Reviewer 說浪費。**只有當有 reranker 才會用到 top 10**。沒 reranker 的話確實 `top_k=10` 是浪費（直接抓 top 4 就好）。連到 #2，修了 #2 自然就用到了。

---

## Reviewer 的 Recommendations 我的看法

| Recommendation | 我的判斷 |
|---|---|
| Mix gold + retrieved during training | **強同意**。修 #1 的最直接路徑。 |
| Add evidence reranker | **強同意**。修 #2 + #19。 |
| Pretrained embeddings (GloVe/fastText) | 同意但**會被 subword (#4 fix) 取代**。如果走 subword 就不需要 GloVe；走 fixed-vocab 才需要 GloVe。 |
| Stratified k-fold | 同意。當前 single dev split 雜訊太大。 |
| Per-class F1 logging | **已經有了**（cell 71 的 `evaluate_cnn_bilstm` 印 per-class F1）。Reviewer 沒看到。 |
| Sanity-check class weights printed | 同意。期望 DISPUTED weight ≈ 4× SUPPORTS weight（因為 train 比例大概 1:4）。 |
| WeightedRandomSampler vs class-weighted CE | 值得試。對小資料集兩個都 valid，效果常常打平，但 sampler 對 batch-level loss spike 比較穩。 |

---

## 我的優先修復順序（不照 reviewer 的）

| # | 改動 | 嚴重度 | 預期收益 | 成本 |
|---|------|--------|----------|------|
| 1 | **#4 Vocab 在全 corpus 建（或換 subword）** | Critical | 最大 | 中（subword 要改架構） |
| 2 | **#11 Best-model 改用 retrieved-dev 選** | Important | 中-大 | 小（多一個 loader） |
| 3 | **#1 Train 加 retrieved hard negatives** | Critical | 大 | 中 |
| 4 | **#2 加 reranker** | Critical | 大（evidence-F） | 中-大 |
| 5 | **#5 BiLSTM 用 pack_padded_sequence** | Important | 1-2 F1 | 小 |
| 6 | **#16 Tokeniser strip trailing punctuation** | Important | 0.5-1 F1 | 極小 |
| 7 | **#3 跑 multi-seed 報 mean±std** | Important | 增加 result 可信度 | 小（重跑幾次） |

---

## 最終 Verdict

**v3 結構正確、實作沒重大 bug、class-balanced loss 會修好 v2 的 F1=0 問題。**

但 v3 解的是 "minority class 全猜不到" 的症狀，沒解到三個更深的瓶頸（vocab UNK、train/predict mismatch、model 不重排）。如果想推到下個檔次，**單做 #4（vocab 全 corpus 或 subword）+ #11（select on retrieved-dev）這兩件事**，預期收益 > 整個 v2→v3 的架構升級。

Multi-kernel CNN + MHA + AttentionPooling 是好東西但邊際遞減；對 1228 個 train claim、6872 vocab 的 setup 來說，input representation 的瓶頸遠比 model capacity 嚴重。

---

*Review 過程：reviewer subagent 給 20 點 → 我對每點做二次 argue → 重新分級 + 排優先序 → 此份紀錄。*
