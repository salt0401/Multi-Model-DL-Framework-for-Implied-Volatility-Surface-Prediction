# 候選方案資料夾

此資料夾保存各 Model 2 候選方案的研究記錄與未來參考資料。

## 方案狀態

| 方案 | 子資料夾 | 狀態 |
|------|---------|------|
| ICNN (B+F) | — (主力，直接在 `src/` 實作) | 🟢 開發中 |
| WamOL PINN (B) | `wamol/` | 🟡 V1 原型用 |
| GNO (E) | `gno/` | 🟡 V3 探索 |
| Heston (C) | `heston/` | ⚪ 存檔 |
| Signature Kernels | `signature/` | ⚪ 存檔 |
| Neural SDE | `neural_sde/` | ⚪ 存檔 |

## 啟動條件

- **WamOL**：V1 階段使用，作為 ICNN 之前的過渡原型
- **GNO**：V2 (ICNN) 成功且產出足夠純淨的歷史 local vol 資料後啟動
- **其餘**：暫不開發，保留 Gemini 研究結論供未來參考
