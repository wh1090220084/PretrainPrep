# 电子书 TXT → 大模型预训练数据集（JSONL 生产基线）

> **文档定位**：这是本仓库的可执行实施说明书，不是只有架构图的概念稿。L0–L5、CLI、审计、断点续跑、JSONL 分片、训练侧随机读取和发布校验均已落地。
>
> **最终训练产物格式：JSONL。** 最终的 `shards/shard_*.jsonl` 每行是一条约定的 `PackedSequence`；分片索引 `shards/index.jsonl` 记录每条 pack 的文件名、字节偏移、字节长度和 token 数。实现不使用 HDF5/Parquet 作为主存储。

---

## 0. 当前交付状态

### 0.1 已实现

- L0：UTF-8/GB18030/Big5/UTF-16 解码、BOM 优先、清洗、页码/目录/页眉页脚处理、硬折行重建、质量向量、多级章节树、异常回退。
- L1：精确 SHA 去重、128-bit SimHash + LSH 近似去重、规则门禁、统计门禁、语言门禁、fastText 适配、auto 启发式回退与降级告警、隔离与审计。
- L2：章节叶子遍历、ids 空间递归切分、字符 span 同步维护、句末/段落边界优先、超长无分隔符硬切、短块回并、无静默丢弃。
- L3：OBFD 装箱、容量桶 Fenwick Tree 加速、EOS 开销计入容量、`position_ids`/`loss_mask`/`doc_lengths` 生成、溢出断言、外部归并排序与分区装箱。
- L4：JSONL 分片、按 token 预算轮转分片、`index.jsonl` 字节偏移索引、`manifest.json`、训练侧 `PackedDataset` 随机读取。
- L5：配置哈希、tokenizer 指纹、阶段依赖哈希、阶段锁、断点标记、审计 JSONL、指标汇总、告警、血缘查询、重复运行字节一致性检查。

### 0.2 未在本工作区验证

- 真实 Qwen/LLaMA tokenizer 的全量数据吞吐。
- fastText 模型启用后的语言判断准确率。
- 1 亿到 10 亿 chunk 规模下的磁盘 I/O、外部排序耗时和内存峰值。
- 200 条人工边界评审、100 本人工清洗评审。
- PyTorch 最小训练脚本的实际 loss 曲线；当前环境未安装 PyTorch，仓库提供可运行脚本但未在本机执行。

---

## 1. 核心承诺与不变量

### 1.1 长度安全

每个最终 pack 在落盘前检查：

```python
len(input_ids) <= bin_size
sum(doc_lengths) == len(input_ids)
len(position_ids) == len(input_ids)
len(loss_mask) == len(input_ids)
len(doc_ids) == len(doc_lengths)
```

L2 启动时还会检查：

```python
l2.max_tokens + len(tokenizer.eos_ids) <= l3.bin_size
```

L3 装箱时以 `n_tokens + len(eos_ids)` 作为真实占用空间，而不是只用 `n_tokens`。任何溢出都会写入 `errors/l3_overflow.json` 并非零退出。

### 1.2 不静默丢弃

- L0 拒绝：写 `quarantine/l0/` 和 `audit/l0.jsonl`。
- L1 拒绝：写 `quarantine/l1/` 和 `audit/l1.jsonl`。
- L1 精确重复输入文件：写 `quarantine/l1/*.alias*.json` 和 `audit/l1_alias_duplicates.jsonl`。
- L2 短块：回并，不按阈值丢弃；整本书低于下限时写 `audit/l2.jsonl` 的 `drop` 决策。
- L3：溢出是 bug，不是正常截断路径，非零退出。
- L4：索引和长度不一致即校验失败。

### 1.3 可追溯性

训练样本中的每个 `doc_id` 可沿以下路径回溯：

```text
PackedSequence.doc_ids
  -> chunks/index.jsonl
  -> chunks/{source_sha}.jsonl 的 TokenizedChunk
  -> chapter_path + char_start/char_end
  -> accepted/{source_sha}.json
  -> normalized/{source_sha}.txt
  -> books/ 原文件 + source_sha256
```

`char_start/char_end` 指向 **NormalizedBook.text**，即清洗后的 UTF-8 文本，而不是原始文件的字节偏移。原始文件通过 `source_sha256` 固定，清洗后文本单独保留，因此可做确定性核验和人工比较。

### 1.4 确定性

同一输入、同一配置、同一 tokenizer、同一代码版本下，最终 `shards/*.jsonl` 和 `shards/index.jsonl` 字节一致。

实现保证：

- 输入文件按相对路径排序。
- 文档按 SHA 排序。
- SimHash 候选按 SHA 排序。
- BFD 按 `(-n_tokens, chunk_id, seq)` 排序。
- 所有 tie-break 显式写出。
- JSON 使用稳定字段顺序和固定分隔符。
- 审计和 manifest 含时间戳，但训练分片不写入运行时间。
- 阶段哈希绑定配置、tokenizer 指纹、上游 manifest 和当前代码指纹。

---

## 2. 总体架构

```text
books/**/*.txt
   │
   ▼
L0 解码 / 清洗 / 质量向量 / 章节树
   ├─ normalized/{sha}.json + {sha}.txt
   ├─ quarantine/l0/
   └─ audit/l0.jsonl
   │
   ▼
L1 SHA 去重 / SimHash-LSH / 规则+统计+语言门禁
   ├─ accepted/{sha}.json
   ├─ dedup/index.jsonl
   ├─ quarantine/l1/
   └─ audit/l1.jsonl
   │
   ▼
L2 ids 空间递归切分 / span / 短块回并
   ├─ chunks/{sha}.jsonl
   ├─ chunks/index.jsonl
   └─ audit/l2.jsonl
   │
   ▼
L3 OBFD packing / position_ids / loss_mask / doc_lengths
   ├─ packed/packed_*.jsonl
   ├─ packed/index.jsonl
   └─ audit/l3.jsonl
   │
   ▼
L4 JSONL 分片 + 字节偏移索引
   ├─ shards/shard_*.jsonl
   ├─ shards/index.jsonl
   └─ shards/manifest.json
   │
   ▼
L5 审计 / metrics / lineage / _SUCCESS / 发布校验
```

---

## 3. 目录结构

```text
Pretrain_Dataset/
├── README.md
├── config.yaml
├── run.py
├── configs/                 # dev/production 配置与 tokenizer 资产
├── src/ebook_pipeline/      # L0-L5 与 JSONL 导出实现
├── examples/                # 样例生成、消费和训练 smoke
├── books/                   # 输入 TXT
├── models/                  # 训练侧 tokenizer 等本地资产
└── data/                    # 运行时生成，不提交
```

`config.yaml` 和 `configs/dev.yaml` 使用 `bytes` tokenizer，只用于冒烟测试。生产必须使用训练侧真实 tokenizer。

---

## 4. 安装

要求 Python 3.9 及以上。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

生产 tokenizer：

```powershell
python -m pip install -e ".[hf]"
```

fastText：

```powershell
python -m pip install -e ".[hf,fasttext]"
```

如果 fastText 在 NumPy 2.x 下报 `np.array(..., copy=False)` 错误，执行：

```bash
python -m pip install "numpy<2"
```

### 4.1 根目录统一启动器

所有平台统一使用 `run.py`，无需手工设置 `PYTHONPATH`：

```bash
python run.py
```

`run.py` 会定位项目根目录、加载 `src`、优先使用项目虚拟环境，并默认读取 `configs/production.yaml`。

常用命令：

```bash
python run.py --force
python run.py validate --sample-packs 100 --sample-chunks 1000
python run.py stage l2 --force
python run.py export-jsonl --output jsonl/pretrain.jsonl --max-tokens 32766 --tokenizer-path <训练侧 tokenizer>
python run.py config-hash
python run.py lineage <pack_id>
```

使用其他配置：

```powershell
$env:EBOOK_PIPELINE_CONFIG = "D:\other\config.yaml"
python run.py
```

---

## 5. 五分钟开发冒烟

### 5.1 生成样例书

```powershell
python examples/make_sample_books.py --output books
```

### 5.2 运行完整 L0–L4

```powershell
python -m ebook_pipeline --config config.yaml run --force
```

运行结束后重点检查：

```text
data-dev/shards/shard_00000.jsonl
data-dev/shards/index.jsonl
data-dev/shards/manifest.json
data-dev/metrics/summary.json
data-dev/audit/l0.jsonl
data-dev/audit/l1.jsonl
data-dev/audit/l2.jsonl
```

### 5.3 读取一条训练 pack

```powershell
python examples/consume_jsonl.py --dataset data-dev/shards
```

### 5.4 查看一个 pack 的血缘

```powershell
python -m ebook_pipeline --config config.yaml lineage pack_p00000_00000000
```

### 5.5 验证分片、偏移和抽样 chunk

```powershell
python -m ebook_pipeline --config config.yaml validate --sample-packs 10 --sample-chunks 1000
```

### 5.6 完整确定性重跑

```powershell
python -m ebook_pipeline --config config.yaml repro-check --alt-work-dir data-dev-repro
```

该命令会在独立工作目录重新跑一次完整流水线，并比较 `shard_*.jsonl` 与 `index.jsonl` 的字节哈希。它可能非常耗时，应在生产发布前执行，而不是每次开发运行。

---

## 6. 生产配置

`configs/production.yaml` 已绑定本目录下的 `tokenizer.json` 和 `tokenizer_config.json`，EOS id 为 `1`。语言检测默认为 `auto`：fastText 和模型可用时使用 fastText，否则回退到启发式并在 summary 中记录 `language_detector_fallback`。如需严格生产模式，设置 `require_fasttext: true`。

configs/production.yaml 的关键字段如下：

```yaml
profile: production

tokenizer:
  backend: tokenizer_json
  name: deepseek-v3-tokenizer       # 必须与训练侧元数据一致
  hf_model: null
  tokenizer_json: tokenizer.json   # 相对 configs/production.yaml
  vocab_files:
    - tokenizer.json
    - tokenizer_config.json
  local_files_only: true
  eos_ids: [1]                     # 当前本地 tokenizer 的 EOS id
  eos_loss: 1

l1:
  language:
    detector: auto
    model_path: ../models/lid.176.bin
    heuristic_fallback: true
    require_fasttext: false

l3:
  bin_size: 32700
  pack_mode: bfd

l4:
  format: jsonl
  shard_size_tokens: 2000000000
```

生产 profile 会强制检查：

- tokenizer backend 不是 `bytes`。
- `tokenizer.name` 非空。
- `tokenizer.vocab_files` 或 `tokenizer.tokenizer_json` 存在，以保证配置哈希绑定词表。
- 当 `require_fasttext: true` 或 `detector: fasttext` 且关闭回退时，fastText 包和模型文件必须存在。
- `l4.format` 为 `jsonl`。
- `l2.max_tokens + EOS 数量 <= l3.bin_size`。

### 6.1 tokenizer 一致性

`config_hash` 的组成：

```text
canonical(config)
+ tokenizer.name
+ tokenizer.vocab_files / tokenizer_json / backend tokenizer 的 SHA-256
```

训练前应校验：

```powershell
python -m ebook_pipeline --config configs/production.yaml config-hash
```

将输出的 `config_hash` 写进训练配置并与 `shards/manifest.json` 比对。tokenizer 不一致时不得训练。

### 6.2 配置文件相对路径

配置中的相对路径相对于 **配置文件所在目录**：

- 根目录 `config.yaml` 的 `books` 指项目根目录下的 `books/`。
- `configs/production.yaml` 使用 `../books`、`../data`、`../models/...`。
- 如果复制配置到其他目录，需要同步调整相对路径或使用绝对路径。

---

## 7. 参数默认值

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `l0.min_chars` | 1024 | 清洗后最小字符数 |
| `l0.replacement_ratio_max` | 0.001 | U+FFFD 硬上限 |
| `l0.control_ratio_max` | 0.005 | 控制字符硬上限 |
| `l1.simhash_bits` | 128 | SimHash 位数 |
| `l1.simhash_hamming` | 3 | 近似重复汉明阈值 |
| `l1.simhash_bands` | 8 | LSH band 数，默认 8 × 16 bit |
| `l1.language.detector` | `auto` | 优先 fastText，缺失时明确回退启发式 |
| `l1.language.require_fasttext` | `false` | `true` 时 fastText 包/模型缺失直接失败 |
| `l1.stat_threshold` | 0.35 | 统计质量分下限 |
| `l1.min_tokens` | 256 | L1 token 下限 |
| `l1.max_url_density` | 0.002 | URL 字符占比上限 |
| `l2.max_tokens` | 30000 | chunk token 硬上限 |
| `l2.min_tokens` | 256 | 短块回并阈值 |
| `l3.bin_size` | 32700 | pack 容量 |
| `l3.max_chunks_in_memory` | 2000000 | 单次 BFD 最大 chunk 引用数 |
| `l3.external_sort_runs` | 64 | 外部归并排序 run 数 |
| `l4.shard_size_tokens` | 2000000000 | 单片 token 预算 |
| `runtime.validation_sample_chunks` | 1000 | 发布抽样 chunk 数 |
| 告警拒绝率 | 40% | `metrics/summary.json` |
| 告警填充率 | 90% | `metrics/summary.json` |
| 告警 cross-doc | 30% | `metrics/summary.json` |

开发配置为了快速运行，使用了更小的 `min_chars`、`max_tokens` 和 `bin_size`，不能用来判断生产填充率。

---

## 8. L0：接入、解码、清洗、结构

### 8.1 解码

`detect_and_decode()` 顺序：

1. BOM：UTF-8、UTF-32、UTF-16。
2. 严格 UTF-8。
3. 无 BOM 候选：GB18030、Big5、UTF-16LE、UTF-16BE。
4. 对每个候选计算 replacement/control/NUL/printable 指标。
5. 通过 replacement 和 control 硬门禁后，按 printable 密度、NUL 惩罚和候选顺序仲裁。

注意：GB18030 和 Big5 存在字节域重叠，任何纯规则方案都有长尾误判。误判样本必须进入隔离区和人工抽检，不应依赖“解码成功”本身作为质量证明。

### 8.2 清洗顺序

严格按以下顺序执行：

1. `\r\n`/`\r` → `\n`。
2. 删除短页码行。
3. 删除目录区间。
4. 保守删除高频短页眉页脚。
5. 重建硬折行，连续合并不超过配置阈值。
6. 连续空行压缩为一个。

页眉页脚删除必须同时满足：

- 出现次数达到阈值。
- 长度不超过 40 字符。
- 不含句末标点。
- 不被章节标题正则识别。

### 8.3 结构树

- 保守模式：`第X卷/章/节/回`、`Chapter N`、`序章/前言/后记/附录`、Markdown `#{1,3}`。
- 宽松模式：`N.`、`一、`、`001 标题` 等。
- 保守标题优先；宽松标题只用于附近没有保守标题的补漏位置。
- 章节数超过 300 或平均字符数过低时，回退为单节“正文”，并写入 `structure_fallback` flag。
- 层级跳跃时生成 `[隐式层级N]` 节点。
- 第一个标题前的正文生成“正文开头”节点，保证叶子覆盖全文。

---

## 9. L1：去重与质量门禁

### 9.1 精确去重

- L0 先按原始文件 SHA-256 分组。
- 同一 SHA 只保留一个规范化文档。
- 其他输入路径写入 `l0_inputs.jsonl` 的 `duplicate_input`。
- L1 为别名路径写 `reason="exact_dup"`, `kept_sha=<canonical sha>`，并写 `audit/l1_alias_duplicates.jsonl`。

### 9.2 SimHash 近似去重

- 128-bit SimHash。
- 默认 4-gram 字符 shingle。
- 默认最多采样 500,000 个 shingle，超长文档使用确定性等距采样。
- 8 band × 16 bit LSH。
- 汉明距离 ≤ 3 判为候选重复。
- 在线 Union-Find 合并近重复组。
- 每组保留 shingle 更多、质量分更高、SHA 字典序更小的文档。

### 9.3 质量门禁

规则层：

- 字符数和 token 数下限。
- URL 密度。
- 广告/联系方式模板。
- 纯标点/纯数字行占比。
- 连续版权声明块长度。

统计层默认公式：

```text
score =
    0.25 * repetition
  + 0.20 * lexical
  + 0.15 * sentence
  + 0.15 * script
  + 0.25 * language
```

其中：

- `repetition = clamp(1 - dup_ngram_ratio / 0.45)`
- `lexical = clamp((vocab_density - 0.05) / 0.35)`
- `sentence` 偏好 8–120 字符的句子。
- `script = clamp((cjk_ratio + latin_ratio) / 0.65)`
- `language = lang_conf`

低于 `stat_threshold` 进入隔离区，不进入切分。

### 9.4 语言

语言检测默认使用 `auto`：优先 fastText `lid.176.bin`，缺失依赖或模型时回退到脚本启发式，并在 `metrics/summary.json` 写入降级告警。语言结果写入：

- `quality.lang`
- `quality.lang_conf`
- `audit/l1.jsonl`
- `metrics/l1.json`

`metrics/l1.json` 同时记录接受稿件的 token 加权语言分布和 `target_language_token_ratio`。

---

## 10. L2：ids 空间语义切分

### 10.1 切分流程

1. 遍历结构树的叶子章节。
2. 取 `NormalizedBook.text[char_start:char_end]`。
3. 按分隔符优先级递归切分：
   `\n\n` → `\n` → `。！？；…` → `.!?;` → `，,`。
4. 某个 part 仍超限且无分隔符时，在字符边界上二分查找最大可编码前缀。
5. 每个 part 单独调用 tokenizer，合并时直接 `ids.extend(part_ids)`。
6. 相邻 part 在不超限时贪心回并，降低碎片。
7. 短块按同章节优先回并；必要时跨章节回并并写 `cross_chapter_merge` flag。
8. 整本书仍低于 `min_tokens` 时写 `drop` 审计；正常 L1 应已拦截。

### 10.2 span 同步

切分函数返回：

```python
(ids, char_start, char_end)
```

每个 chunk 的 `char_end` 是最后一个 part 的字符结束位置。span 连续、无重叠，且覆盖该章节叶子中的非空内容。

### 10.3 目标长度

不强制每个 chunk 都达到 30,000 token：

- 整节 ≤ `max_tokens`：整节一块。
- 超长节：优先自然边界递归切。
- 无自然边界的超长段：硬切。
- 不按固定长度凑块。

---

## 11. L3：OBFD Packing

### 11.1 算法

1. 将所有 chunk 按 `n_tokens` 降序排序。
2. tie-break：`chunk_id`、`seq`。
3. 容量桶 Fenwick Tree 记录所有 bin 的剩余容量。
4. 对每个 chunk 找“剩余空间 ≥ chunk_size 且剩余最小”的 bin。
5. 找到则放入并更新容量；否则新建 bin。
6. bin 内最终按 `seq` 排序写盘，保留书内顺序。
7. `chunk_size = n_tokens + len(eos_ids)`，EOS 空间不会被漏算。

### 11.2 大规模路径

- 默认 `max_chunks_in_memory=2,000,000`。
- 超过阈值时，先构建 chunk 索引，再用外部归并排序生成有序 run。
- 按 `max_chunks_in_memory` 切分 deterministic partition，每个 partition 独立 BFD。
- 该路径会牺牲少量跨分区填充率，但避免把全部 chunk token 放进内存。
- `pack_mode=per_book` 用于长上下文连续性实验；当前实现要求整批 chunk 引用可驻留内存，否则报错。

### 11.3 训练语义

```python
input_ids = concat([chunk.ids + eos_ids for chunk in pack])
doc_lengths = [len(chunk.ids) + len(eos_ids) for chunk in pack]
position_ids = concat([list(range(d)) for d in doc_lengths])
loss_mask = concat([[1] * len(chunk.ids) + [eos_loss] * len(eos_ids) for chunk in pack])
```

- `position_ids` 每个文档从 0 重置。
- `loss_mask` 默认 EOS 也计损失。
- `cross_doc` 按不同 `source_sha256` 判断，不按 chunk 数判断。
- 不截断超长 chunk；超长直接失败。

---

## 12. L4：最终 JSONL 分片

### 12.1 分片策略

- 格式：未压缩 JSONL。
- 默认 `shard_size_tokens = 2,000,000,000`。
- 每条 PackedSequence 一行。
- 一个 pack 不跨 shard。
- 当加入下一条 pack 会超过 shard token 预算时轮转新 shard。
- 分片写入使用临时文件加原子替换，崩溃时最多丢失正在写入的 shard。

### 12.2 索引

`shards/index.jsonl` 每条记录：

```json
{"pack_id":"pack_p00000_00000000","shard_id":0,"file":"shard_00000.jsonl","offset":0,"byte_length":158421,"n_tokens":32700,"n_docs":3,"cross_doc":true,"source_shas":["...","..."]}
```

`offset` 和 `byte_length` 是 UTF-8 字节偏移，不是字符数。

### 12.3 训练读取

```python
from ebook_pipeline.dataset import PackedDataset

with PackedDataset("data-dev/shards") as ds:
    pack = ds[ds.rows[0]["pack_id"]]
    print(len(pack["input_ids"]), pack["doc_lengths"])
```

读取路径：

1. 内存加载 `index.jsonl`。
2. 按 `pack_id` 查行。
3. seek 到对应 shard 的字节偏移。
4. 读取并解析一条 JSON。
5. 返回 Python list；训练 collator 再转 tensor。

这是一个最小解析路径，但不是零解析。选用 JSONL 意味着每次随机读取需要解析一条 JSON；如果未来需要极低随机读取开销，可以增加 HDF5/Arrow 二级缓存，但这不改变主产物格式。

---

### 12.4 通用 JSONL 文本导出

`export-jsonl` 会先自动执行完整流水线；已完成的阶段会跳过，缺失阶段会自动补齐。需要强制重算时加 `--force`。

当训练框架要求在训练时自行 tokenizer，而不是直接消费 `input_ids` 时，可以导出通用文本 JSONL，每行：

```json
{"text":"一段预训练文本"}
```

导出命令：

```powershell
python .\run.py export-jsonl `
  --output jsonl/pretrain.jsonl `
  --max-tokens 32766 `
  --tokenizer-path <训练侧 tokenizer 目录>
```

输出：

```text
D:\研究项目\Pretrain_Dataset\data\jsonl\pretrain.jsonl
```

导出规则：

- 每行只有 `text` 字段，不包含内部 `input_ids`。
- `--max-tokens` 是训练侧 tokenizer 下的单行正文上限。
- `--tokenizer-path` 应指向训练侧 tokenizer；省略时使用本项目配置中的 tokenizer。
- 导出器会按 token 预算重新切分文本，不先生成超长行再交给训练侧截断。
- 导出统计写入 `data/metrics/jsonl_export.json`。

以训练框架 `max_seq_len=32768`、额外添加 BOS/EOS 为例，正文预算为：

```text
32768 - 2 = 32766
```

训练框架示例：

```powershell
python trainer/train_pretrain.py `
  --data_path D:\研究项目\Pretrain_Dataset\data\jsonl\pretrain.jsonl `
  --max_seq_len 32768
```

如果其他训练框架的最大上下文不同，按 `max_tokens = 最大上下文 - 框架添加的特殊 token 数` 计算。

---

## 13. JSONL / JSON Schema

### 13.1 TokenizedChunk

`chunks/{sha}.jsonl` 每行：

```json
{"chunk_id":"abcdef123456_00000000","seq":0,"source_sha256":"abcdef...","book_title":"第一章 开始","chapter_path":["第一卷","第一章 开始"],"chapter_level":1,"char_start":0,"char_end":1234,"ids":[12,34,56],"n_tokens":3,"flags":[]}
```

字段说明：

| 字段 | 含义 |
|---|---|
| `chunk_id` | `sha256 前 12 位 + 全局 seq` |
| `seq` | 全数据集单调递增，BFD 内部排序键 |
| `source_sha256` | 源文件内容哈希 |
| `book_title` | 从章节结构推导的书名/首标题 |
| `chapter_path` | 从根到叶的章节路径 |
| `chapter_level` | 叶子层级 |
| `char_start/char_end` | 在 `NormalizedBook.text` 中的字符 span |
| `ids` | 不含 EOS 的 token ids |
| `n_tokens` | `len(ids)` |
| `flags` | 回并等审计标记 |

### 13.2 PackedSequence

`shards/shard_*.jsonl` 每行：

```json
{"pack_id":"pack_p00000_00000000","input_ids":[1,2,3,256,4,5,6,256],"doc_lengths":[4,4],"position_ids":[0,1,2,3,0,1,2,3],"loss_mask":[1,1,1,1,1,1,1,1],"doc_ids":["abcdef123456_00000000","abcdef123456_00000001"],"source_sha256s":["abcdef..."],"fill_ratio":0.000244,"cross_doc":false}
```

| 字段 | 训练用途 |
|---|---|
| `input_ids` | 模型输入；已含每段 EOS |
| `doc_lengths` | 文档边界；Megatron `cu_seqlens`、block-diagonal mask |
| `position_ids` | 每文档从 0 重置 |
| `loss_mask` | 1=参与 loss；EOS 按 `eos_loss` |
| `doc_ids` | 血缘查询 |
| `source_sha256s` | 源文件集合 |
| `fill_ratio` | `len(input_ids) / bin_size` |
| `cross_doc` | 是否包含多个源文件 |

### 13.3 审计 JSONL

`audit/l1.jsonl` 示例：

```json
{"sha256":"...","stage":"l1","decision":"reject","reason":"url_density","score":0.018,"config_hash":"..."}
```

`audit/l2.jsonl` 额外包含：

```json
{"sha256":"...","stage":"l2","decision":"accept","reason":"","score":1.0,"chunks":42,"tokens":180000,"min_chunk_tokens":256,"mean_chunk_tokens":4285.7,"max_chunk_tokens":29871,"dropped_chunks":0}
```

### 13.4 manifest

`shards/manifest.json`：

```json
{
  "shard_count": 12,
  "total_tokens": 23100000000,
  "total_packs": 706422,
  "bin_size": 32700,
  "tokenizer": "Qwen/Qwen2.5-7B-Instruct",
  "config_hash": "...",
  "format": "jsonl",
  "shards": [
    {"shard_id":0,"path":"shards/shard_00000.jsonl","tokens":1999999998,"sha256":"...","bytes":18321000000}
  ],
  "created_at": "2026-10-29T12:00:00+00:00"
}
```

`created_at` 是运行元数据；分片字节本身不依赖时间戳。

---

## 14. 训练侧对接

### 14.1 基本读取

```python
from ebook_pipeline.dataset import PackedDataset

with PackedDataset("data/shards") as ds:
    for row in ds.rows:
        pack = ds[row["pack_id"]]
        # input_ids, position_ids, loss_mask, doc_lengths
```

### 14.2 Megatron

```python
cu_seqlens = [0]
for length in pack["doc_lengths"]:
    cu_seqlens.append(cu_seqlens[-1] + length)

# 配合 --reset-attention-mask --reset-position-ids
# cu_seqlens 用于变长序列和文档边界
```

### 14.3 block-diagonal attention mask

```python
def block_diag(doc_lengths):
    size = sum(doc_lengths)
    mask = [[0] * size for _ in range(size)]
    cursor = 0
    for length in doc_lengths:
        for i in range(cursor, cursor + length):
            for j in range(cursor, cursor + length):
                mask[i][j] = 1
        cursor += length
    return mask
```

`examples/consume_jsonl.py` 提供了同逻辑的最小示例。

### 14.4 训练冒烟

```powershell
python examples/train_smoke.py --dataset data/shards --vocab-size 32000 --steps 100
```

该脚本只验证“能读取、能切窗、能按 loss_mask 训练、无 shape 错误”。它不是生产训练脚本，也不证明模型质量。

---

## 15. 断点续跑

标记目录：

```text
state/l0/{sha}.done
state/l1/{sha}.done
state/l2/{sha}.done
state/l3/run.done
state/l4/run.done
```

规则：

- 标记内容包含 `stage_hash`。
- `stage_hash` 绑定当前阶段配置、tokenizer 指纹、上游 manifest 和代码指纹。
- 配置或代码变化会自动使旧标记失效。
- L0/L1/L2 按 SHA 跳过已完成文档。
- L3/L4 是全局阶段，使用 run 级标记。
- 强制重算：

```powershell
python -m ebook_pipeline --config config.yaml run --force
```

或单阶段：

```powershell
python -m ebook_pipeline --config config.yaml stage l2 --force
```

注意：`--force` 会重新生成当前阶段和后续可触发阶段的输出，但不会自动删除无关的旧文件。下游始终以 manifest 中列出的活动文件为准，避免把历史文件误读进数据集。

---

## 16. 并发控制

每层使用 `state/locks/{stage}.lock`：

- 同一阶段不允许两个进程同时写。
- 锁文件含 PID、主机名、创建时间。
- 不要手工删除正在运行任务的锁。
- 只有确认进程退出后才可删除遗留锁。

示例：

```powershell
Get-Content data-dev/state/locks/l1.lock
```

---

## 17. 指标与告警

`metrics/summary.json` 提供：

- 漏斗：输入文档 → L0 通过 → L1 通过 → chunks → packs → tokens。
- token 加权填充率。
- 拒绝原因 top-10。
- chunk 长度直方图。
- cross-doc pack 比例。
- 语言分布。
- 质量分分布。
- 告警列表。

默认告警：

| 指标 | 阈值 |
|---|---:|
| 拒绝率 | > 40% |
| 填充率 | < 90% |
| cross-doc 比例 | > 30% |

出现 `status="review_required"` 时应人工复核，不应直接发布。

---

## 18. 发布验收

| # | 检查项 | 本仓库命令/证据 | 是否自动阻断 |
|---|---|---|---|
| 1 | 溢出检查 | `validate` 检查所有 pack 长度 | 是 |
| 2 | 可还原性 | `validate --sample-chunks 1000` 检查 decode/span | 是（抽样范围） |
| 3 | 确定性 | `repro-check` 比较 shard/index 字节哈希 | 是 |
| 4 | 人工边界/章节抽样 | 需 200 chunk 人工评审；`chunk_validation.json` 提供边界比例 | 否，人工门槛 |
| 5 | 去重有效性 | `chunk_validation.json` 的 `cross_doc_8gram_repeat_ratio` + 1000 chunk 抽样复核 | 部分自动 |
| 6 | 训练侧消费 | `examples/train_smoke.py` 跑 100 步 | 外部验证 |
| 7 | 血缘完整性 | `validate` 抽样 + `lineage <pack_id>` | 是（抽样范围） |
| 8 | 语种纯净度 | `metrics/l1.json` 的 `target_language_token_ratio` | 人工/发布检查 |

推荐发布流程：

```powershell
# 1. 全量构建
python -m ebook_pipeline --config configs/production.yaml run

# 2. 结构校验
python -m ebook_pipeline --config configs/production.yaml validate --sample-packs 100 --sample-chunks 1000

# 3. 确定性重跑
python -m ebook_pipeline --config configs/production.yaml repro-check --alt-work-dir D:\data\repro

# 4. 训练冒烟
python examples/train_smoke.py --dataset data/shards --vocab-size <vocab_size> --steps 100

# 5. 人工抽样 200 chunk + 100 文档清洗 diff
# 6. 确认 target_language_token_ratio > 0.95
# 7. 检查 summary.json 无阻断告警
```

任何自动检查失败，或用人工检查超过阈值，都应阻塞发布。修复后从失败层续跑；如果 schema、tokenizer 或清洗规则变化，应视为新版本并重新生成 manifest。

---

## 19. 失败与隔离

| 目录/文件 | 含义 |
|---|---|
| `quarantine/l0/` | 编码/长度/控制字符失败 |
| `quarantine/l1/` | 规则/统计/语言/近似重复失败 |
| `quarantine/l1/*.alias*.json` | 精确重复输入文件 |
| `errors/l3_overflow.json` | L3 溢出或装箱 invariant 失败 |
| `audit/l*.jsonl` | 每层决策记录 |
| `metrics/*.json` | 每层指标 |

拒绝记录必须包含 `stage`、`reason`、`score`、`config_hash`。不允许只写日志文本而不写结构化审计。

---

## 20. 性能与规模边界

### 20.1 当前实现适合的边界

- L0/L1/L2：按 SHA 串行处理，天然可扩展到多进程；当前 CLI 为单进程基线。
- SimHash：默认每文档一个 128-bit 整数和少量元数据，100 万文档量级可接受；1000 万以上建议磁盘 LSH。
- L1 Union-Find：近重复组和 LSH 桶常驻内存。重复极多的合集数据会显著增加内存。
- L3：默认 200 万 chunk 引用驻留；超过后走外部排序和分区 BFD。
- JSONL：便于流式读取和人工检查；每行解析比 HDF5 hyperslab 慢，但兼容性最好。
- 未启用压缩。若需要压缩，必须保留独立字节索引或改用可 seek 的容器。

### 20.2 TB 级建议

1. 先按 SHA 前缀把书分到多个 worker，L0/L1/L2 并行。
2. L2 完成后合并 `chunks/index.jsonl`。
3. L3 使用低内存机器或单独大内存机器，调整 `max_chunks_in_memory`。
4. 先用 1 万本书做端到端压测，记录：
   - 每分钟处理书籍数；
   - chunk 平均 token；
   - L3 峰值 RSS；
   - 外部排序临时磁盘占用；
   - 最终填充率；
   - 单 shard 写入吞吐。
5. 如果填充率低于 90%，优先增大 `max_chunks_in_memory` 或降低分区数，而不是放宽长度不变量。

### 20.3 已知风险

- 章节识别长尾：保守/宽松双模式仍可能漏章或把正文短行误判为章节。必须做人工抽样。
- 编码歧义：GB18030/Big5 重叠区可能误判。隔离区和 `encoding_detected` 可用于复核。
- SimHash 假阳性：近重复阈值 3 较严格，仍建议对隔离组做抽样。
- 短块回并：跨章节回并会改变章节路径边界，已通过 flag 和审计显式记录。
- 页眉页脚规则：过于激进会删正文；规则已保持保守，但上线前必须做 diff 抽样。
- JSONL 随机读取：每次读 pack 都需要 JSON 解析，训练侧应使用批量顺序读取或本地页缓存。

---

## 21. 安全、合规与数据治理

- 不自动下载 tokenizer、fastText 模型或书籍；由部署方提供并校验许可证。
- `trust_remote_code=False`，避免 Hugging Face tokenizer 加载时执行远程代码。
- 原始书籍、清洗文本、审计记录可能含版权或个人信息，应按组织的数据分级制度访问。
- 不要把 `data/`、`books/`、模型文件或审计记录提交到公共仓库。
- 生产发布前确认数据授权、版权状态、去重策略和语言范围符合法律与业务要求。

---

## 22. 常用命令

```powershell
# 初始化目录
python -m ebook_pipeline --config config.yaml init

# 全量运行
python -m ebook_pipeline --config config.yaml run

# 强制全量重算
python -m ebook_pipeline --config config.yaml run --force

# 单阶段
python -m ebook_pipeline --config config.yaml stage l0 --force
python -m ebook_pipeline --config config.yaml stage l1 --force
python -m ebook_pipeline --config config.yaml stage l2 --force
python -m ebook_pipeline --config config.yaml stage l3 --force
python -m ebook_pipeline --config config.yaml stage l4 --force

# 校验
python -m ebook_pipeline --config config.yaml validate --sample-packs 100 --sample-chunks 1000

# 确定性重跑
python -m ebook_pipeline --config config.yaml repro-check --alt-work-dir data-repro

# 血缘
python -m ebook_pipeline --config config.yaml lineage <pack_id>

# 配置哈希
python -m ebook_pipeline --config configs/production.yaml config-hash

# 重建 summary
python -m ebook_pipeline --config config.yaml summary
```

---

## 23. 本地检查

当前工作区未包含 `tests/`。可执行以下无数据依赖的语法检查：

```bash
python -m compileall -q src examples run.py
```

完整端到端检查需要 `books/`、训练侧 tokenizer 和已生成的 `data/state/l1/manifest.json`。

---

## 24. API 速查

```python
from ebook_pipeline.config import load_config
from ebook_pipeline.pipeline import Pipeline
from ebook_pipeline.dataset import PackedDataset
from ebook_pipeline.validation import validate_dataset, validate_chunks, inspect_lineage

config = load_config("configs/production.yaml")
result = Pipeline(config).run_all(validate=True)

with PackedDataset("data/shards") as ds:
    pack = ds[ds.rows[0]["pack_id"]]

report = validate_dataset(config, sample_packs=100)
chunk_report = validate_chunks(config, tokenizer, sample_size=1000)
lineage = inspect_lineage(config, pack["pack_id"])
```

---

## 25. 设计取舍

| 决策 | 选择 | 理由 |
|---|---|---|
| 最终主存储 | JSONL + 字节索引 | 用户要求 JSONL；可流式、可读、可 diff、跨框架 |
| 去重时机 | L1，切分前 | 保留上下文，避免切分后重复破坏 |
| 切分空间 | ids 空间 | 长度精确可加 |
| EOS | id 列表 | 多 token EOS 安全 |
| Packing | OBFD + 容量桶 Fenwick | 填充率与规模化平衡 |
| 短块 | 回并 + audit | 避免静默丢尾块 |
| 配置绑定 | 配置 + tokenizer + 上游 manifest + 代码指纹 | 防止旧标记误用 |
| 高召回章节 | 保守优先 + 宽松补漏 | 兼顾精度和召回 |
| 语言门禁 | fastText 优先，auto 模式显式降级并告警 | 兼顾可运行性与生产可观测性 |

---

## 26. 后续可扩展项

- 多进程 L0/L1/L2 worker。
- 磁盘版 SimHash-LSH。
- 段落级 MinHash-LSH。
- 质量模型插件。
- 训练侧课程式长度采样。
- JSONL 压缩与独立 chunk 索引。
- 数据看板与自动告警通知。

这些扩展不应改变以下契约：最终训练样本仍是可验证的 PackedSequence JSONL；每个 token 仍可回到 source SHA、章节路径和 char span。
