# PIVoTS

PIVoTS 研究時間序列中「每個工具區塊都不足以識別完整因果參數」的問題：先建立有效的動態 IV moments，再選擇區塊、估計局部投影與整合資訊。

目前版本 **v0.2.0**：已完成動態識別修正、可重現基準、付費 pilot 原型，以及真正無 pilot Random 的成本／日曆期限對照。900次初步實驗未顯示pilot普遍優於Random；在刻意設計的方向重複情境，pilot資訊選擇提高了母體識別成功率。一般一致性／選擇後推論定理、完整性能研究與異步資料方法仍待完成。

## 環境

新版採平面配置，從 repository 根目錄執行，沒有 `src/` 子目錄。測試環境為 Python 3.12。可使用現有 `PIVoTS_env`，或建立新環境：

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` 固定已測試的直接依賴版本，並非所有平台的完整環境鎖檔。核心流程不需要 CVXPY、Kaleido 或 SLURM。

## 先驗證識別

```sh
python -m unittest discover -s tests -v
python validate_identification.py --output validation/identification.json
```

檢查包含：精確平穩母體 moment、錯誤控制的負向對照、局部投影恆等式、累積 rank，以及 12 次 replication、樣本長度 300/1500/6000 的收斂實驗。JSON 保存每次誤差與 seed；已存在的輸出不會被覆蓋。

## 有限預算基準：noisy oracle

```sh
python run_optimization.py --fig_path Simulation_Results/smoke \
  --n_runs 3 --n 300 --p 3 --d 6 --d_id 3 --d_max 2 --n_rounds 2 --budget 4
```

此入口刻意保留真實 first-stage 加噪音的 selection features，作為 **noisy-oracle 基準**。不能把它當成可由實際資料直接取得的 selector。

Proposed 與 Random 共用資料、每輪工具數與有限總預算。結果包括單一區塊、選中工具聯集、全部工具 IV、OLS；Full IV 是可能超預算的參考，不能當作同成本競爭者。

每個目錄包含 `manifest.json`、`results.npy`、標明 `reference_only_not_a_trial` 的示例資料。Manifest 記錄全部設定、DGP、seed、工作檔雜湊、套件版本及完成狀態。改參數請使用新目錄；明確指定 `--overwrite` 才會替換既有結果。

## 可行選擇原型：pilot → 選集合 → 後段估計

```sh
python run_pilot_selection.py --output Simulation_Results/pilot_smoke \
  --n_runs 3 --n_pilot 150 --n_main 500 --gap 100 \
  --p 3 --d 6 --block_size 2 --instrument_budget 4
```

所有候選先取得短期 pilot。正式觀測前凍結選擇日程；選擇只使用 pilot moments 與 joint HAC covariance，不讀取真實係數或後段資料。比較資訊增益、Random、強度優先，並在相同選中資料上比較 GMM 與等權投影整合。另列全工具 GMM 參考。

本原型使用相等大小區塊；每塊維度小於 `p`，`d` 與 `instrument_budget` 必須可被 `block_size` 整除。它研究「已付費取得共同 pilot 後的追加觀測」，不宣稱節省所有來源開通費。成本包含 IV 當期與所需歷史：pilot `(n_pilot+1)*d`，正式資料 `(n_main+1)*selected_IV_count`。共同 X/Y 成本不計入比較。模擬器生成未選資料供參考基準使用，但不提供給 selector。

Pilot 與正式資料取自同一條時間序列的不同區段，中間留 `gap`。有限 gap 不代表獨立；尚未建立選擇後信賴區間。樣本 rank 亦不能證明母體強識別。Rank 不足的 trial 明確回報投影代表解，其完整參數誤差留空；摘要標明條件樣本，配對比較使用共同完整 rank 的 trial。

## 無 pilot Random：主要成本基準

```sh
python run_selection_benchmark.py --output Simulation_Results/no_pilot_smoke \
  --n_runs 3 --n_pilot 60 --n_main 80 --gap 30 \
  --p 3 --d 6 --block_size 2 --instrument_budget 4 \
  --dgp_seeds 81 --scenarios dense heterogeneous redundant
```

比較五種策略：無pilot Random（相同n／近似同總成本）、pilot Random、pilot資訊選擇、pilot強度選擇。真正無pilot的策略在看資料前抽取區塊、pilot成本為0，以自己的正式資料估計。省下的pilot成本可以增加正式樣本，但不可超過共同觀測期限。

成本模型假設整段X/Y已經存在，只收取新增IV觀測成本。正式工具數上限不限制pilot暫時觀測全部候選。結果分開報告母體欠識別、樣本rank、數值失敗、條件RMSE、尾部與配對MSE差；不得只憑成功trial的誤差判斷整體優劣。

輸出包括先於模擬固定的 `protocol.json`、含每個trial的 `benchmark_results.json`、逐DGP的兩份摘要CSV。新的runner拒絕覆寫結果；不與舊schema的報表混用。此次900次實驗與完整指令見 [無pilot比較紀錄](docs/BENCHMARK_NO_PILOT.md)。

## 圖表與正式網格

```sh
python make_simulation_tables.py \
  --result_path Simulation_Results/smoke/p3_d6_d_id3_d_max2_n_rounds2 \
  --save_path Simulation_Tables/smoke
python plot_simulation_summary.py \
  --result_path Simulation_Results/smoke/p3_d6_d_id3_d_max2_n_rounds2 \
  --save_path Simulation_Figures/smoke
```

表格列出有限結果數、失敗數、MCSE、p95/p99 及配對差異。聚焦圖之外，同時输出主實驗完整範圍圖及失敗摘要 CSV。只接受完成的 schema-v2 基準結果；pilot 結果使用獨立 JSON，不能混入 noisy-oracle 圖表。

`run_simulation_grid.py` 與 `run_rho_only.py` 共用設定及目前的 Python interpreter。預設為每情境 500 次 replication，請先跑小型驗證，再明確啟動正式網格。`run_optimization_slurm.py` 共用主 CLI；叢集提交範本的環境路徑仍需依實際叢集配置。

## 識別範圍與結果相容性

- 對齊 `Y[t]`、`X[t-1]`、`I_selected[t-2]`，以常數和 `I_selected[t-3]` 殘差化。
- 模擬 validator 採取充分条件：選中工具區塊自主演化、工具沒有直接 H/Y 路徑、IV 與非 IV innovations 獨立、穩定 Gaussian VAR(1)。不符合時拒絕以這組控制宣稱有效。
- rho 預設只改變對角持續性，固定 beta、first-stage 和混淆路徑係數。平穩變異與有效精度仍可能改變。
- 模擬由 Lyapunov 平穩共變異數初始化。`legacy_rescale` 只保留舊的整體縮放方式，不會還原舊的錯誤控制或舊隨機樣本。
- 數值 SVD rank、方向 coverage、HAC 和 ridge 都不能證明工具外生性，也不能把未識別參數變成已識別。
- 舊論文圖表必須重新生成；本次修改後的數值不能與旧版本直接合併。
- `run_simulation_grid_overlap.py` 已明確標為尚未實作。舊 `visualization.py`、`plot_test_fig1.py` 與 finite-sample/cluster scripts 屬 legacy，不是新研究路線的正式報告入口。

詳見 [方法與研究進度](docs/METHODOLOGY.md) 及 [驗證紀錄](docs/VALIDATION.md)。


## 版本管理規則

每完成一個研究階段，先通過對應測試並記錄實驗的限制，再更新本節、建立 Git commit 與版本標籤，推送至 GitHub。研究開發使用 `codex/research-stages` 分支；尚未完成的工作不標示為已驗證。原始大量模擬輸出不提交，提交可重現指令、設定與精簡驗證紀錄。

## 版本變更

### v0.2.0 — 2026-10-11：無 pilot Random 與成本／期限對照

- 新增真正零pilot成本的Random與節省成本增加樣本的基準，共五個比較arms。
- 加入共同日曆期限、預算餘額、母體識別診斷、獨立失敗記錄及逐DGP配對統計。
- 固定dense／強弱不均／重複方向三情境、三個結構seed，每組100次，共900次；全部符合預算與期限。
- 47項本機測試通過。一般與強弱不均情境未見穩定pilot優勢；重複方向情境pilot資訊選擇完整識別96%，Random38–39%，條件RMSE不能替代識別成功率。
- 提交重現設定、程式雜湊與精簡結果；原始trial輸出保留本機。詳見 [實驗紀錄與限制](docs/BENCHMARK_NO_PILOT.md)。
- 下一步：pilot長度／預算／工具強度／近似重複的敏感度，以及選擇後理論；尚未完成一般推論或性能優越性證明。

### v0.1.0 — 2026-10-11：識別修正、可重現基準與 pilot 原型

- 修正時間對齊與 IV 歷史控制，加入受限 VAR 模型的母體 moments、局部投影檢查及平穩初始化。
- 修正有限預算、配對 trial、結果保存與統計報告；明確標記 noisy-oracle 比較。
- 加入付費 pilot、joint HAC GMM 與資訊增益選擇原型，區分樣本 rank 與母體識別。
- 本機 29 項測試通過；12 次 replication 的收斂檢查通過。詳見 [驗證紀錄](docs/VALIDATION.md)。
- 已知限制：5 次 pilot 小型實驗沒有優於 Random；當時 Random 同樣支付 pilot，尚未檢验無 pilot 策略。一般理論、選擇後推論與正式大規模評估未完成。
