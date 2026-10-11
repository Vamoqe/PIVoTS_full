# 方法、假設與尚待完成的工作

## 第一階段：已落實的受限模型

狀態順序為 `(I, X, H, Y)`，scalar Y，dim(H)=dim(X)。目標是結構方程中 X[t-1] 對 Y[t] 的係數，並非未加區分的長期總效果。

對選中區塊 m，假設 I_m[t]=R_m I_m[t-1]+nu_m[t]，不含區塊外父節點。Innovations 跨時間 iid Gaussian，IV 與非 IV innovations 獨立；I 不直接進入 H 或 Y 方程。每塊可以使用自己的 I_m[t-3] 歷史。

控制 B=I_m[t-3] 後，Z=I_m[t-2] 的母體殘差為 nu_m[t-2]。它與 Y[t]-beta'X[t-1] 中的 H[t-1]、Y[t-1] 及 outcome innovation 正交：這些項沒有從該新 instrument innovation 到達的允許路徑。因此 E[Z_tilde(Y_tilde-X_tilde'beta)]=0。所有變數使用相同 B 殘差化。此論證依賴指定結構，單純 OLS 殘差化不能替代結構假設。

令 G_m=E[Z_tilde X_tilde']，h_m=E[Z_tilde Y_tilde]。局部 minimum-norm 解識別 P_row(G_m) beta；指定 Q_m beta 可識別需要 row(Q_m) 包含於 row(G_m)。不施加額外結構時，完整 beta 的識別需要堆疊 G 滿 column rank。

對每塊 full row rank、保留全部 moment 且使用完整聯合共變異數的可逆正交化，加權投影整合與相應 GMM 等價。不同權重或丟棄方向時不能直接套用此等價；本版等權投影是獨立對照。

## 第二階段：已落實的比較契約

同一個 DGP 固定結構係數，每個 trial 重新抽取時間序列。方法共用該 trial 資料。Noisy-oracle 主流程對 proposed/random 固定同一 block-size schedule 和總預算，保存 trial ID、seed、各輪分組與實際成本。

跨 rho 固定因果/first-stage/混淆係數，只調整對角 autoregression。這不固定全部平穩共變異數；不能把研究結果解釋為所有其他統計性質完全不變。

## 第三階段：已實作但尚待性能研究的 pilot 原型

固定有限 p、K 與共同時間尺度。先付費觀測全部候選的短 pilot，再凍結逐步選擇日程，在不重疊的後段資料區間估計；不假定兩區段在機率上獨立。Pilot 必須提供可用的初始識別。X 的標準化由 pilot 決定並對全部方法保持一致，最後還原 coefficient 單位。

從 pilot 估計 G、h、初始 beta 和完整 joint Bartlett HAC covariance Omega。資訊分數使用 J_S=G_S' Omega_SS^{-1} G_S 的 regularized logdet 邊际增量除以追加成本。Random 和強度優先共享付費 pilot、候選區塊與總預算。Covariance eigenvalue floor 為 max(1e-10*lambda_max,1e-12)，結果保存 regularization 診斷；selection logdet ridge=1e-6，不是最終 beta 的 ridge。

資訊準則是可執行的候選 heuristic，不宣稱一般 submodularity、全域最優或必然在預算內找出滿 rank 集合。Full sample rank 不等於強識別。所有推論目前限於點估計；沒有後選擇信賴區間。

## 第四階段：無 pilot 的成本與日曆期限對照

新增 `run_selection_benchmark.py`，保留原本 paid-pilot runner 作為歷史對照。五個 arms 為無 pilot Random（相同正式樣本數）、無 pilot Random（利用總預算增加樣本）、pilot Random、pilot information、pilot strength。

所有 Random arms 在資料產生前用獨立 selection seed 抽取同一組區塊。無 pilot 方法從時間列 0 開始正式觀測，尺度只用自身正式 X；pilot 方法用前段 pilot 的尺度與 moments，正式區間從 `n_pilot+3+gap` 開始。有限 gap 不被當成獨立性的證明。

總日曆期限為 `H=n_pilot+n_main+gap+6` 列。每個選中工具需要 `n+1` 列 IV／IV 歷史，總預算為 `B=(n_pilot+1)*d+(n_main+1)*k`，其中 k 是正式選中工具數。無 pilot 的追加樣本數為 `min(floor(B/k)-1,H-3)`；完整記錄餘額，不把離散成本或期限上限造成的未使用預算說成完全花完。

**成本適用範圍：** 整個 H 期間的 X/Y 是已存在且各方法均可使用的資料，只比較新增 IV 觀測成本。不同樣本數的 X/Y 若需要付費，此成本模型不適用。正式工具數上限是 k；pilot 暫時觀測全部 d 個候選，因此不是「任何時間最多只能觀測 k 個工具」的硬體上限設計，也未計算來源開通費。

預先固定三種 first-stage 結構，其他結構係數保持相同；每個 DGP seed 之下，三種情境的 first-stage Frobenius norm 相同：

1. `dense`：原本同幅度範圍的隨機正負係數。
2. `heterogeneous`：區塊強度因子由 0.15 到 1 的等比序列，排列由 seed 固定。
3. `redundant`：區塊 1 複製區塊 0 的 first-stage 係數，再調整整體尺度；其餘區塊保留。候選聯集仍必須母體滿秩。

真實母體 moments 只用於情境驗證和選擇後評估，沒有傳入 selector。除了樣本 rank，亦回報選中區塊的母體 rank、原始參數單位下 G 的最小奇異值與 condition number。後者是模型診斷，不是實務資料可取得的 IV 有效性測試。母體或樣本 rank 不足時，完整 beta 的誤差為空值，並且明列識別失敗數。

每個固定 DGP 的重複抽樣分別摘要，主比較為相對無 pilot、近似同總成本 Random 的配對 squared-error 差與 MCSE；另保存對 pilot Random 的比較。配對誤差只使用雙方有效的共同 trials，明列排除數。這是條件比較，不能替代識別成功率。不同 DGP 的結果不當成同一固定分布來計算 MCSE。各 selector 的失敗獨立記錄，共同 pilot 初始化失敗則同時影響依賴它的 selectors。

## 尚待完成的研究

1. 在固定有限候選、適当 mixing/矩條件、pilot/main 成長速度、時間間隔、HAC 一致性與分數間隔等條件下，完整證明選擇後一致性；需要漸近常態時再處理有效變異數。
2. 檢查噪音、尺度、局部弱工具和全域近共線對分數品質的影響；比較小 K 下的可行集合枚舉。
3. 多個結構 seed、預算、pilot 長度、工具強度與 rho 的預先規劃評估，報告配對差異、尾部和失敗。
4. 異步資料必須保留共同 causal lag，處理觀測指標和缺漏機制，並與 missing-aware GMM 比較。
5. 交通/天氣仍為資料結構示例；實證因果解釋需要額外的排除限制論證。

不要把數值檢查、prototype 或這份條件說明寫成已完成的一般定理或新穎性證明。

## 參考基礎

- Ailer, Hartford & Kilbertus (2023): https://proceedings.mlr.press/v202/ailer23a.html
- Thams et al. (2024), particularly Theorem 6: https://www.jmlr.org/papers/v25/22-0262.html
- Ailer et al. (2024): https://arxiv.org/abs/2405.19985
