# 情绪标签打标方案

## 当前可复现结果

正式数据按歌曲 ID 去重后有 5,043 首、5,894 条原始标签记录。固定留出
20% 作为测试集，得到 4,034 首训练歌和 1,009 首测试歌，其中 601 首是
单标签样本，408 首是多标签样本。训练集和测试集按歌曲 ID 完全隔离。

当前模型使用字符级 TF-IDF（`char_wb`、1–5 gram、最多 150,000 特征），
将歌曲名、艺人、一级曲风重复 5 次后拼接歌词，再使用
`OneVsRest(LinearSVC(C=0.3, class_weight=balanced))`。SVC margin 在服务边界
转换为行 softmax，保证置信度落在 `[0,1]` 且排序与模型一致。

在固定留出集上：

| 指标 | 结果 |
| --- | ---: |
| 单标签 Top-1 | 60.90% |
| 宏平均召回 | 56.06% |
| 单标签 Top-2 候选覆盖 | 77.54% |
| 单标签 Top-3 候选覆盖 | 85.52% |
| 单标签 Top-7 候选覆盖 | 95.67% |
| 全部样本 Top-3 候选覆盖 | 87.31% |

这组数据说明提升准确率的关键是对相邻标签做候选消歧，而不是把更多
标签同时返回。服务仍只输出一个最置信的二级标签，Top-2 仅作为复核候选。

## 达标运行链路

1. 请求进入同步 API，校验 15 个正式标签所需的字段。
2. 本地 TF-IDF/SVC 生成 Top-7 候选、置信度和 margin。
3. margin 足够大时直接放行，减少外部模型调用；歌词、标题、艺人和风格
   缺失时只使用真实存在的字段。
4. 低 margin（默认分差 < 0.10）样本送入内网语义复核器，提示词中只允许从 Top-7 候选中选一类，
   同时传入标签定义和实际歌词片段。复核器必须返回候选标签、置信度和证据，
   服务严格校验后才覆盖本地结果。
5. 复核器不可用或返回非法标签时回退本地结果，并在 evidence 中记录实际使用
   的输入；不伪造音频或外部事实。
6. 将低 margin、模型与复核器不一致的样本写入人工复核队列。人工结果进入
   新版本训练集，按固定测试集重新评测，通过后原子切换 bundle 指针。

### EmoCompass 方法落地

本服务已经把参赛方案中最有价值的四个环节做成了可运行组件：

- `competition_service/rubric/emotion_rubric.json` 是版本化 Rubric。它覆盖全部
  15 个正式标签，提供定义、正向线索、反例和唯一规则 ID；当前以
  `soft_context` 传给语义复核器，只帮助候选消歧，不把同极性标签硬性排除。
- 语义复核响应必须带有可核验的歌词引文。服务会检查引文确实出现在本次输入
  歌词中，并检查返回的 Rubric 规则 ID；校验失败时安全回退本地模型。
- `python -m competition_emotion.calibration` 从固定 Official Test 的 Train
  内部再切出 Dev/校准集，默认种子为 `20260911`、比例为 15%。校准集只用于
  阈值、路由和规则调参，不能改写固定 Test。
- `iteration.py` 提供 JSONL trace、低 margin/复核分歧困难样本挖掘、Rubric
  patch 提案和回归门禁。提案状态固定为 `proposed`，必须经过人工复核和门禁
  才能进入下一版 Rubric，不会由线上请求自动改生产规则。

启动服务时可设置 `EMOTION_RUBRIC_PATH` 和 `EMOTION_TRACE_PATH`，或分别传入
`--rubric-path`、`--trace-path`。trace 不记录完整音频，只保存模型版本、Rubric
版本、候选、分差、复核结果和已验证证据，适合按日离线挖掘。

示例：

```powershell
py -3.12 -m competition_emotion.calibration `
  --workbook F:\netease\_music\competition\data\emotion_songs_20260910.xlsx `
  --output F:\netease\_music\competition\runs\official-20260910\calibration-20260911.json
```

离线迭代顺序是：读取 trace → `mine_hard_cases` → 生成 proposed patch → 在 Dev
和固定 Test 上评测 → `evaluate_patch_gate` 通过后人工确认 → 发布新的不可变
Rubric/model bundle。

```powershell
py -3.12 -m competition_emotion.iteration `
  --trace-path F:\netease\_music\competition\runs\official-20260910\traces\emotion.jsonl `
  --output F:\netease\_music\competition\runs\official-20260910\rubric-patch-proposal.json
```

通过 Dev/Test 门禁并完成人工复核后，调用 `publish_rubric_candidate` 原子替换
目标 Rubric 文件；未显式批准或门禁失败时会拒绝发布。

Top-7 覆盖率是语义复核阶段的上限参考，不等于最终准确率；只有在内网复核器
完成独立留出集验证后，才能声明达到 80% 宏召回门槛。

## 成本策略

本地模型常驻内存，单次调用不需要 GPU，适合数万首/月的批量生产。音频下载和
解码仍受独立并发上限及 25 秒总预算控制，且未经验证的基础音频特征不参与当前
标签排序。语义复核仅处理低 margin 样本；上线前用留出集统计放行比例、复核比例、
P50/P99 延迟和每千首成本，再选择 margin 阈值。任何阈值调整都必须使用固定测试
集之外的新校准集，避免把评测集用于调参。

## 可扩展和迭代方式

标签定义、标签顺序和训练配置集中在 `constants.py` 与 `models.py`，接入新标签
只需补充定义、训练样本和评测集，训练器会检查正负样本是否齐全。模型、报告和
预测结果以不可变版本目录发布，`current.json` 只做一次原子切换；失败训练不会
污染线上版本。报告保留源文件 SHA-256、切分种子、样本数和各标签召回，便于回归
比较和定位劣化。
