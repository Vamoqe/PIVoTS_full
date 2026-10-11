# 驗證紀錄（2026-10-11）

本紀錄對應此次識別／基準修正與 pilot 原型，環境為 macOS arm64、Python 3.12.11、requirements.txt 所列直接依賴。它不構成一般定理或正式性能評估。

## 自動化測試

`python -B -m unittest discover -s tests -v`：29 項通過。

涵蓋：有效母體 moment、錯誤控制的負向對照、局部投影、rho 保持結構係數、未支援 IV 路徑拒絕、對齊、VAR(1)/VAR(2) 平穩初始化、SVD 子空間與 n<p、零 coverage、IID 入口、budget/BaseSet、主流程重現性、每 trial 筆數、manifest、覆寫保護、單維度／單輪軌跡、joint HAC、資訊增量、欠識別標示、pilot 成本及非有限結果統計。

另實際從 CLI 執行兩次 replication 的主流程，成功輸出 CSV、LaTeX、四組主圖 PNG/PDF（含完整範圍），沒有啟動正式 500 次 grid。

## 母體與大樣本檢查

設定 p=3、d=6、seed=253、rho=0.8，其他為新版預設；使用 12 個已記錄的 innovation seed。不同樣本量使用同一 trial 路徑的前綴，避免把不同 DGP 混成收斂趨勢。

- 正確 I-history 的母體 moment norm：5.19e-17。
- 錯誤 X-history 的母體 moment norm：0.02722。
- 全體 moment matrix rank：3。
- 局部 projection identity 通過。

| 樣本數 | Full IV RMSE | 固定低維區塊投影整合 RMSE |
|---:|---:|---:|
| 300 | 0.32260 | 1.02152 |
| 1,500 | 0.13391 | 0.41258 |
| 6,000 | 0.05576 | 0.18597 |

兩種估計在這個檢查中都隨樣本增加而改善。投影整合沒有因此超越 Full IV；這些結果驗證指定模型下的實作方向，不是泛化定理。完整 trial 資料位於 [validation_identification.json](validation_identification.json)。

## Pilot 原型的小規模結果

預設 p=10、d=15、block_size=3、instrument_budget=12、pilot=300、main=1000、gap=100、seed=253，5 次 replication：

| 方法 | 平均 L2 error | 含 pilot 的總成本 |
|---|---:|---:|
| 資訊增益選擇 | 0.44681 | 16,527 |
| Random | 0.35000 | 16,527 |
| 強度優先 | 0.40982 | 16,527 |
| 全工具 GMM 參考 | 0.25049 | 19,530 |

全部五次均為 sample full rank。新 selector 在這次小樣本檢查中沒有勝過 Random；這個原型需要評估 pilot 長度、HAC 誤差、DGP 和強度條件，不能先宣稱效能提升。只有五次 replication，不能據此判定普遍優劣。Full GMM 超過同成本比較的預算，亦不是有限樣本誤差的數學上界。

## 尚未驗證

- 正式多 DGP、多成本設定的 Monte Carlo。
- 選擇後漸近推論與信賴區間。
- 非平穩、異步／informative missingness，以及交通／天氣工具有效性。
- 新建 GitHub Actions 尚未在遠端執行；本機測試已通過。


## v0.2.0：無 pilot 對照（同日新增）

本機完整測試增至47項，全數通過。新增900次固定情境／trial的成本與日曆期限對照，結果與限制見 [BENCHMARK_NO_PILOT.md](BENCHMARK_NO_PILOT.md)。上方5次pilot檢查是v0.1.0的歷史紀錄，並非目前唯一的實驗證據；兩個runner的預算、時間區間與Random定義不同，不直接合併。
