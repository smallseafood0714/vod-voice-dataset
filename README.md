# vod-voice-dataset

把自己的 YouTube / Twitch VOD 清洗成語音模型（TTS）可用的訓練資料。
這個 repo 只負責**資料清洗**；下載與訓練另外處理。

## 流程

每一步讀前一個資料夾、寫下一個資料夾，中間產物都保留，方便回頭檢查哪一步出問題。

| 步驟 | 輸入 → 輸出 | 做什麼 |
|---|---|---|
| 0 | 手動 → `data/00_raw/` | 放手動下載的影片或音訊（優先挑沒 BGM、沒別人聲音的） |
| 1 | `00_raw` → `01_audio/` | 用 ffmpeg 抽音軌、轉成單聲道 WAV |
| 2 | `01_audio` → `02_vocals/` | 人聲分離，去掉殘留的音樂與遊戲音效 |
| 3 | `02_vocals` → `03_segments/` | 用 VAD 切成 3～15 秒小段 |
| 4 | `03_segments` → `04_filtered/` | 說話人比對（只留自己的聲音），並篩掉爆音、重疊、太短的段落 |
| 5 | `04_filtered` → `05_dataset/` | 轉文字，輸出最終的音檔與文字對照清單 |

`ref/` 放 1～2 分鐘自己的乾淨錄音，第 4 步用來當聲紋基準。

## 環境

- Windows 11、Python 3.12、ffmpeg
- GPU：RTX 4060 Laptop（8 GB VRAM）

音訊資料一律不進 git（見 `.gitignore`）。
