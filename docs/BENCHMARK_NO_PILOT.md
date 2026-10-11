# 無 pilot 對照：v0.2.0 驗證紀錄

日期：2026-10-11。這是初步、預先固定設定的 Monte Carlo 對照，並非一般優越性證明或投稿所需的完整性能研究。

## 可重現設定

```sh
python run_selection_benchmark.py --output Simulation_Results/no_pilot_v0.2.0 \
  --n_runs 100 --dgp_seeds 253 517 809 --scenarios dense heterogeneous redundant
```

若上述輸出已存在，請選新目錄；程式拒絕覆寫。其餘參數為固定預設：p=10、d=15、block_size=3、正式工具數12、pilot300、main1000、gap100、rho0.8。

3 種情境 × 3 個結構 seed × 每組100次，共900個情境／trial，每次比較5個方法。結構與 seed 在觀察結果前固定於 `protocol.json`。補強 selector 錯誤隔離後用相同 seed 重跑，正常案例結果逐項相同；重跑不計為額外獨立 replication。此次沒有調參搜尋或依結果刪除情境。

本機測試：47項通過，涵蓋資料隔離、成本／期限、可重現性、母體欠識別、selector 獨立失敗與配對統計。900次實驗所有 arms 均符合預算與共同日曆期限，沒有數值失敗；方向重複情境有實際母體欠識別，另行計數。

## 成本與比較範圍

| 方法 | Pilot成本 | 正式樣本 | 正式成本 | 總成本 |
|---|---:|---:|---:|---:|
| 無 pilot Random，相同 n | 0 | 1000 | 12012 | 12012 |
| 無 pilot Random，近似同總成本 | 0 | 1376 | 16524 | 16524 |
| Pilot Random / information / strength | 4515 | 1000 | 12012 | 16527 |

所有方法的共同日曆期限為1406列，無 pilot 方法從列0開始，pilot正式區間從列403開始。三種Random共用選中集合，但估計區间可不同。3單位的預算差是完整時間列的離散成本，未強行花掉。

**前提：整段 X/Y 已可取得，只計額外IV量測成本。** Pilot可暫時量測全部15個候選，正式區間限制12個；若任何時間都只能量測12個，這個pilot配置不可行。本實驗也未計來源開通費。

## 一般與強弱不均情境

下表均有100/100個可評估trial。主比較為 pilot information 相對近似同總成本、無pilot Random。RMSE以原始beta單位計算；配對MSE差為 pilot 減 Random，負值有利於pilot。MCSE描述此固定DGP下有限Monte Carlo的精度，不是因果參數的標準誤，也未作多重比較顯著性宣稱。

| 情境 | DGP seed | Random RMSE | Pilot資訊 RMSE | 配對MSE差 | MCSE |
|---|---:|---:|---:|---:|---:|
| dense | 253 | 0.4465 | 0.5319 | 0.08355 | 0.12888 |
| dense | 517 | 0.4292 | 0.3625 | -0.05278 | 0.04422 |
| dense | 809 | 0.3018 | 0.3256 | 0.01490 | 0.01814 |
| heterogeneous | 253 | 0.7495 | 0.6062 | -0.19436 | 0.14548 |
| heterogeneous | 517 | 0.6384 | 0.7176 | 0.10750 | 0.09380 |
| heterogeneous | 809 | 0.5290 | 0.3922 | -0.12604 | 0.04451 |

這六組設定中，資訊選擇的點估計結果有好有壞，沒有穩定超越無pilot Random。部分MSE差的MCSE很大，100次亦不足以精確刻畫重尾誤差。不能將某個seed的改善解讀成普遍優勢。

## 重複方向：先看識別成功率

此情境刻意令區塊0和1擁有相同first-stage方向。五選四時，Random只有省略其中一個重複區塊才可能完整識別，這三個固定DGP的枚舉結果亦確認只有2/5個可行集合滿秩，因此理論成功率40%。它是一個壓力測試，沒有聲稱現實工具必然有這種結構。

| DGP seed | Random完整識別 | Pilot資訊完整識別 | Pilot強度完整識別 | Random條件RMSE | Pilot資訊條件RMSE | 共同有效配對 |
|---|---:|---:|---:|---:|---:|---:|
| 253 | 38/100 | 96/100 | 63/100 | 0.3153 | 0.6048 | 36/100 |
| 517 | 39/100 | 96/100 | 43/100 | 0.3264 | 0.3620 | 38/100 |
| 809 | 38/100 | 96/100 | 37/100 | 0.2349 | 0.3228 | 38/100 |

這次pilot資訊選擇提高了完整識別的頻率，但自身仍有4%的欠識別trial。兩方法條件RMSE的母集合不同，Random的低誤差只反映其38–39個成功案例，不能用來推論全體trial的優劣。資料中即使母體rank不足，有限樣本moment matrix仍可能數值滿秩；新版會把這些trial明確排除完整beta誤差並計入識別失敗。

## 研究決策與下一階段

目前保留無pilot Random作主要成本基準，pilot作具備條件性價值的選擇模組。值得後續研究的問題是「方向重複／弱識別何時讓pilot的資訊值得其成本」，而非宣稱pilot總是必要或Random總是足夠。

接下來固定評估不同pilot長度、正式預算、rho、工具強度與近似重複程度，並增加結構seed與replication；同時處理識別成功率、最弱方向與條件誤差。理論上仍欠一致性／選擇後推論與相對既有投影整合／GMM的明確新增貢獻。本階段未建立信賴區間或一般貪婪最優保證。

完整五方法的逐DGP摘要、尾端誤差、排除數與配對MCSE位於 [method_summary.csv](benchmarks/v0.2.0/method_summary.csv)、[paired_comparisons.csv](benchmarks/v0.2.0/paired_comparisons.csv)、[summary.json](benchmarks/v0.2.0/summary.json)。[protocol.json](benchmarks/v0.2.0/protocol.json) 保留全部設定及程式SHA256。逐trial原始結果保存在本機 `Simulation_Results/no_pilot_v0.2.0/benchmark_results.json`，依專案規則不加入Git。
