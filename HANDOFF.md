# 交班不变量说明

本文说明 `report_robust` 的判定口径、固定字段与父链，供下一班按已发布标识核对。
业务接口与调用格式仍以 README.md、ROBUST.md 为准。

## 判定口径

- 冻结空间中的实单位轴 `u` 由拟合视图的 `basis` 张成；基向量可以非正交，投影按欧氏内积计算，方向正负视为同一轴。
- 每个位置用本位置的修订、门槛和剔除预算。同一修订可在左右两个位置复用，也可与另一修订组合。
- 位置门槛 `t`、剔除预算 `d`、行数 `n`（1～12）满足 `0 <= d < n`。读数重复出现时按行位置分别计数。
- 位置在轴 `u` 上保留 `k = n - d` 行时的平方投影均值为
  `sum_{i in S} (c_i . u)^2 / k`，其中 `c_i` 是扣除冻结训练中心后的行坐标，`S` 为任一 `k` 行留样。
- 位置的“最坏留样均值”是全部 `S` 上该均值的最小值；它随 `k` 增大不减，因此预算只允许剔除不超过 `d` 行。
- 判定为真当且仅当存在同一根 `u`，使两个位置的最坏留样均值同时达到各自门槛；门槛含等号。
- 一维冻结空间直接比较唯一方向；二维冻结空间在射影直线上取有限个二次型的全部实零点与每段开弧内的一个有理点逐点核验，零点保留为 `a + b*sqrt(R)` 的精确形式，故无理方向与孤立达标点不被遗漏。

## 固定字段与读回

`read_result(archive, 'report', report_id)` 对两类报告都返回 V1 字段
（`kind, report_id, fit_id, train_rev, left_rev, right_rev, unit, energy_unit, center,
lo_left, lo_right, passed`），有理量以 `分子/正分母` 字符串表示。

新报告另含整数 `drop_left`、`drop_right`，其 `energy` 为
`worst_retained_mean_squared_projection_about_training_center`。
V1 报告保持 `energy = mean_squared_projection_about_training_center`，编码与字段值不变。

预算、门槛、结论和父链一并写入报告体与关联表；其他输入和结论相同而预算不同的两份报告各有独立标识。读回只解码已发布内容，不重算。

## 父链与更正

- 报告固定 `fit:` 标识、训练修订、左右修订；更正登记为新修订并使用 `parent_rev` 指向原修订，原报告不变。
- 报告发布是单一事务：报告体与关联任一失败（含 `before_commit` 抛出）整体回滚，随后可重试。
- 非法绑定、超限行数或非法预算在写事务前被拒绝，归档行数不变。

## 交班核验清单

1. 用 `.pointwise-runtime.json` 的 `verify` 命令运行 `python3.11 -B -m robust_examples verify`：先跑完整原回归，再核对 ROBUST.md 手算结论、非法输入与回滚、更正重开后的固定读回。
2. 用 `demo` 命令运行 `python3.11 -B -m demo demo`：查看逐对可行却无共同轴、逐位置预算读回、更正重开后的固定字段。
3. 按已发布 `report:` 标识读回，确认 `fit_id / train_rev / left_rev / right_rev / lo_left / lo_right / drop_left / drop_right / passed` 与发布时一致。
