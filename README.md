# vod-voice-dataset

把自己的 YouTube / Twitch VOD 清洗成語音模型（TTS）可用的訓練資料。
這個 repo 只負責**資料清洗**；下載與訓練另外處理。

## 流程

每一步讀前一個資料夾、寫下一個資料夾，中間產物都保留，方便回頭檢查哪一步出問題。

| 步驟 | 輸入 → 輸出 | 做什麼 |
|---|---|---|
| 0 | 手動 → `data/00_raw/` | 放手動下載的影片或音訊（優先挑沒 BGM、沒別人聲音的） |
| 1 | `00_raw` → `01_audio/` | 用 ffmpeg 抽音軌，轉成 44.1 kHz 雙聲道 WAV（分離模型要吃雙聲道） |
| 2（選用） | `01_audio` → `02_vocals/` | 人聲分離，去掉殘留的音樂與遊戲音效，輸出單聲道。乾淨的畫圖實況可跳過 |
| 3 | `02_vocals` 或 `01_audio` → `03_segments/` | 用 Silero VAD 切成 2～15 秒小段，附 `segments.csv` |
| 4 | `03_segments` → `04_filtered/` | 說話人比對（只留自己的聲音），並篩掉太小聲、爆音的段落，附 `scores.csv` |
| 5 | `04_filtered` → `05_dataset/` | 轉文字，輸出最終的音檔與文字對照清單 |

`ref/` 放 1～2 分鐘自己的乾淨錄音，第 4 步用來當聲紋基準。

## 資料管理

音訊不進 git，git 只管「怎麼重建資料」：

- `sources.csv`（repo 根目錄，進 git）：每部影片一列，記影片 ID、網址、日期、標題、備註。
  新影片下載後加一列。`00_raw/` 弄丟了可以照這份重新下載，之後的步驟都能用腳本重跑。
  存成 UTF-8 with BOM，Excel 直接開不會亂碼。
- `data/` 底下的中間產物都可以重建，不另外備份。
- 要保留的是重建不回來的東西：`ref/` 的錄音，和最後的 `05_dataset/`（轉文字可能手動修過）。
  這兩個另外備份到雲端硬碟或外接硬碟。

各步驟的輸出檔名都是影片 ID（yt-dlp 檔名裡 `[...]` 的部分），沒有 ID 的檔案就用整理過的檔名。
每支腳本都會跳過已經有輸出的檔案，所以新影片放進 `00_raw/` 之後重跑整串即可。

## 環境

- Windows 11、Python 3.12、ffmpeg（要在 PATH 上）
- GPU：RTX 4060 Laptop（8 GB VRAM）

安裝（第一次）：

```bash
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

模型權重第一次執行時會自動下載到 `models/`。

## 執行

```bash
.venv\Scripts\python.exe scripts\01_extract_audio.py
.venv\Scripts\python.exe scripts\03_segment.py
.venv\Scripts\python.exe scripts\04_filter.py
```

需要人聲分離的影片（遊戲、音樂），在第 1 步和第 3 步之間加跑，後面接影片 ID。
不加 ID 會把 `01_audio/` 裡所有還沒分離的影片都跑一遍：

```bash
.venv\Scripts\python.exe scripts\02_separate_vocals.py --cpu <影片ID> <影片ID>
.venv\Scripts\python.exe scripts\02_check_excerpts.py
```

- 第 2 步預設用 `vocals_mel_band_roformer.ckpt`（Kim 的 MelBand RoFormer），可用 `--model` 換。
- **目前一定要加 `--cpu`**：這台筆電的 RTX 4060 一跑分離就會掉卡，幾分鐘後藍屏（0x113）。
  `--cpu` 會把顯示卡藏起來完全不碰。CPU 速度約等於即時（1 小時有聲音訊要 1 小時）。
- 低於 `--threshold-db`（預設 -50 dB）的部分不送進模型，直接輸出靜音，輸出和原檔時間軸對齊。
  實況裡沒講話時幾乎是數位靜音（-75 dB 以下），這樣只需要跑 15～35% 的時間。
  背景一直有遊戲聲或音樂的影片幾乎整段都超過門檻，就省不到時間。
- 有聲音的片段會打包成最多 10 分鐘一批送進模型；超過 10 分鐘的連續片段會切開、重疊 5 秒淡入淡出接回去。
  記憶體用量不隨檔案長度增加。
- `02_check_excerpts.py` 會挑出「人聲和被去掉的聲音都很大」的段落，
  把前後對照各 30 秒存到 `data/02_vocals/_check/`，用來試聽分離效果。
- **第 2 步是選用的**：試聽過兩部畫圖實況，分離前後幾乎一樣（被去掉的聲音比人聲小 15～23 dB），
  所以畫圖實況直接從第 1 步接第 3 步。只有遊戲聲或音樂明顯的影片才需要跑第 2 步。
- 第 3 步用 Silero VAD（只用 CPU）找出講話的位置，切成 2～15 秒的單聲道小段，
  輸出到 `data/03_segments/<影片ID>/`，檔名是 `<影片ID>_<開始毫秒>.wav`。
  有 `02_vocals/<影片ID>.wav` 就用分離後的人聲，沒有就用 `01_audio/` 的原音。
  停頓不到 0.3 秒不切；兩段之間停頓不到 0.8 秒、合起來不超過 15 秒就併成一段；短於 `--min-sec`（預設 2 秒）的丟掉。
  每部影片有一份 `segments.csv`，記錄每段的起訖時間、長度、音量（`rms_db`）、峰值（`peak_db`）、
  爆音取樣比例（`clip_pct`）和來源（`source`：`02_vocals` 或 `01_audio`）。
  實況音訊被限幅器壓在接近 0 dB，峰值 0 dB 不代表爆音，要看 `clip_pct`。
  （最早的 `q_BUeuHkOT0`、`4yOK4xE075Q` 兩部是用 3 秒門檻、分離後的人聲切的，CSV 沒有 `source` 欄。）
- 第 4 步用 SpeechBrain 的 ECAPA 聲紋模型（只用 CPU）算每段的聲紋，跟「你的聲音」比相似度。
  `ref/` 裡有 wav 就用它當基準；沒有就用全部片段聲紋的平均（大部分片段都是你，平均就是你），
  再用最像的 80% 重算一次平均。相似度低於 `--min-sim`、音量低於 `--min-rms-db`（-35 dB）、
  爆音比例高於 `--max-clip-pct`（0.1%）的片段丟掉，其餘複製到 `data/04_filtered/<影片ID>/`。
  每段的分數和丟掉的原因寫在 `data/04_filtered/scores.csv`。
  聲紋會快取在 `data/04_filtered/_cache/`，新影片進來重跑時只會算新的；
  基準是用全部片段重算的，所以舊影片的去留也可能跟著微調。

音訊資料一律不進 git（見 `.gitignore`）。
