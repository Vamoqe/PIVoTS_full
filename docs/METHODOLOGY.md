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
