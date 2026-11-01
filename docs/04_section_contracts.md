# Section contracts

## Section: 研究背景与问题
- Purpose: 说明多研究异质性为什么使方法验证成为独立科学问题。
- Inputs: 01 canon、02 evidence table。
- Allowed claims: 现有研究间方差、稀疏性和口径敏感性。
- Forbidden claims: 作物因果效应、真实网络相互作用。
- Required evidence: 45 项目摘要统计和前三篇结果。
- Validation checklist: 每个数字可回溯；不重复论文 1–3 的结论。

## Section: 数据基础与仿真标定
- Purpose: 将真实数据压缩为可检验的仿真参数。
- Inputs: feature matrix、study/platform metadata、depth、prevalence。
- Allowed claims: 参数分布、项目随机效应和不确定性。
- Forbidden claims: 用单点估计代表全部研究。
- Required evidence: 数据审计、拟合诊断、参数 bootstrap。
- Validation checklist: 假数据摘要落在真实数据预设区间；不通过不得进入基准。

## Section: 仿真场景
- Purpose: 预注册生物学效应与技术扰动网格。
- Inputs: 项目数、样本数、深度、零比例、区域、参考库偏移、效应量。
- Allowed claims: 场景覆盖与计算预算。
- Forbidden claims: 用场景数量代替科学合理性。
- Required evidence: 场景表、随机种子、运行清单。
- Validation checklist: 场景不按结果临时增删；主/敏感性场景分开。

## Section: 方法基准
- Purpose: 规定待比较方法、输入和禁止使用真值信息。
- Inputs: 现有论文方法、候选统计工具。
- Allowed claims: 方法类别和预期失败模式。
- Forbidden claims: 依据结果随意更换主方法。
- Required evidence: 版本、参数搜索空间、嵌套验证。
- Validation checklist: 所有调参仅在训练/内层折完成。

## Section: 主终点与统计判据
- Purpose: 定义方法胜负不以单一指标决定。
- Inputs: FDR、功效、偏倚、覆盖、core F1、网络 AUPRC、分类校准。
- Allowed claims: 预注册主终点和共同主排序。
- Forbidden claims: 事后选择最好指标。
- Required evidence: 指标公式、聚合层、置信区间。
- Validation checklist: 项目/平台层聚合；报告不确定性。

## Section: 结果、建议与边界
- Purpose: 输出按场景可执行的方法选择规则。
- Inputs: outputs、robustness runs。
- Allowed claims: 在已测场景中的相对表现。
- Forbidden claims: 无限外推。
- Required evidence: 主结果、敏感性、真实数据一致性。
- Validation checklist: 方法排名方向在敏感性中保留或明确降级。

