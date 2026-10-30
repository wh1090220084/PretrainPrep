# 电子书预训练数据流水线实施计划

## 授权与目标

- 目标：在 `D:\研究项目\Pretrain_Dataset` 中实现 TXT -> JSONL 预训练数据集的完整、可运行、可测试流水线。
- 用户已明确提出“按照方案打造生产级方案”和“输出格式是 jsonl”，因此最终分片、索引、审计和指标均采用 JSONL/JSON，不引入 HDF5/Parquet 主存储。
- 本次交付范围：L0 归一化、L1 去重与门禁、L2 ids 空间切分、L3 OBFD packing、L4 JSONL 分片与随机读取、L5 编排/审计/断点续跑/校验、测试与详细 README。

## 明确排除

- 不实现 PDF/EPUB/MOBI 转换、OCR、SFT 数据构造、质量模型训练。
- 不下载 tokenizer、fastText 模型或训练数据。生产运行需要调用方提供模型与词汇表文件。
- 不承诺在本机完成 TB 级压测；通过流式 JSONL、外部排序和分片边界控制提供可扩展实现，并记录容量边界。

## 关键接口与数据流

1. L0 读取 `books/**/*.txt`，输出 `normalized/*.json + normalized/*.txt`、隔离样本和审计记录。
2. L1 先执行可并行的质量门禁，再对候选做确定性 SHA/SimHash 去重；输出 `accepted/*.json`、`dedup/index.jsonl`。
3. L2 在 token ids 空间切分，同时维护字符 span；输出 `chunks/*.jsonl` 和 chunk 索引。
4. L3 读取 chunk 索引，用容量桶 Segment Tree 实现 OBFD；超大批次使用确定性外部归并排序和分区装箱。
5. L4 将 packed JSONL 重组为 `shards/shard_*.jsonl`，生成带字节偏移的 `shards/index.jsonl` 和 manifest。
6. L5 提供 CLI、阶段锁、配置哈希、断点标记、审计汇总、指标、血缘查询和发布校验。

## 不变量

- 每个 packed 序列满足：`len(input_ids) <= bin_size`、`sum(doc_lengths) == len(input_ids)`、`len(position_ids) == len(input_ids)`、`len(loss_mask) == len(input_ids)`。
- 所有被拒绝文档或块均有结构化 `reason/score/stage`。
- 所有排序使用完整 tie-break；输出 JSON 使用稳定字段顺序和稳定分隔符。
- `config_hash` 由规范化配置和 tokenizer 指纹共同决定，写入每个阶段的 manifest 和审计。
- 同一输入、配置、tokenizer 和代码版本重复执行，分片文件哈希一致。

## 验证计划

- 使用 `unittest` 覆盖编码、清洗、结构识别、门禁、SimHash、切分 span、OBFD、JSONL 索引和端到端流程。
- 执行完整测试套件。
- 用小样本执行 L0-L4 两次，比较最终 shard 哈希，验证确定性。
- 执行发布校验器，检查 schema、长度、索引偏移、chunk 回溯和源文件 SHA。
- README 记录已验证证据、未验证的 TB 级容量、外部模型依赖和生产上线前检查项。

## 回滚与风险

- 代码均为新增文件；回滚可删除本仓库新增内容。
- 数据输出写入配置指定工作目录，已有输出由阶段 manifest 驱动消费；强制重建时先写临时文件再原子替换。
- 主要残余风险：章节识别长尾召回、无 tokenizer 时的开发模式不适用于生产、fastText 模型缺失导致生产 profile 校验失败、TB 级外部排序和磁盘 I/O 未现场压测。