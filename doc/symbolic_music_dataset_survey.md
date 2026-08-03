# 符号音乐数据集调研：面向 Jazz Arranger 的 Harmony 视角

> 目标：不是先把训练任务定义做得很完美，而是先找出**能服务“旋律 → 较复杂爵士和声编排”**的数据。
> 结论先行：**HookTheory/HLSD、POP909、EMOPIA+** 是最直接的 harmony 监督源；**PDMX** 适合做公有领域谱面预训练；**MetaMIDI / GigaMIDI / Discover / Godzilla / Lakh** 更适合做大规模弱监督预训练，再自己做 chord extraction。

## 1. 快速结论

- **最适合直接学 harmony**：HookTheory / HLSD、POP909、EMOPIA+。
- **可从谱面或功能表示里恢复 harmony**：PDMX、EMOPIA（原始版可自己抽 lead sheet / chord）。
- **适合大规模预训练，但原生没有统一和弦真值**：MetaMIDI、GigaMIDI、Discover、Godzilla、LMD、Slakh、MAESTRO、MetaScore、XMIDI、社区 anime MIDI。
- **对 jazz arranger 来说的优先顺序**：
  1. HookTheory/HLSD
  2. POP909
  3. EMOPIA+
  4. PDMX
  5. MetaMIDI / GigaMIDI / Discover / Godzilla / LMD 做弱监督和风格扩充

## 2. 统一口径

- **规模**：尽量使用公开 README / paper / dataset card 的数字；若不同版本有差异，会在备注里说明。
- **Harmony 标记**：
  - **显式**：人工和弦、lead sheet、功能和声、和弦事件。
  - **可恢复**：MusicXML / lead sheet / 结构化标记里可以解析出和声，但不是统一 chord label 表。
  - **无统一标注**：需要自己跑 chord recognition / harmony extraction。

---

## 3. 数据集逐个说明

### 3.1 PDMX

- **链接**：
  - demo: https://pnlong.github.io/PDMX.demo/
  - code: https://github.com/pnlong/PDMX/
  - data: https://zenodo.org/records/15571083
- **格式**：MusicXML; project internal processing also uses MusicRender JSON.
- **规模**：>250K public-domain scores; paper also reports 102,635 deduplicated unique arrangements.
- **覆盖范围**：Public-domain repertoire, with classical and folk strongest; top-10 also includes rock, pop, jazz, and electronic. 67% of songs have no genre tag.
- **Harmony 情况**：Recoverable, but not a unified chord-label dataset.
- **样本结构**：One work -> one MusicXML / MusicRender JSON; common metadata: title, composer, genre, rating, track count.
- **评论**：Best as a legal, score-level pretraining base. Good for structure and voice-leading, not a ready-made jazz harmony corpus.

### 3.2 MetaMIDI

- **链接**：
  - repo: https://github.com/Metacreation-Lab/MetaMIDI-Dataset
  - Zenodo: https://zenodo.org/record/5142664
- **格式**：MIDI + structured metadata (JSONL / TSV / ID mappings in the repo).
- **规模**：436,631 MIDIs; 221,504 with artist/title; 143,868 with genre.
- **覆盖范围**：Broad Western popular music / film / game / mixed styles, with strong metadata and audio links.
- **Harmony 情况**：No unified chord labels.
- **样本结构**：md5 / title / artist / genre / Spotify IDs / MusicBrainz IDs.
- **评论**：Useful for style filtering and weak supervision. Treat it as a big corpus, not a chord-truth set.

### 3.3 GigaMIDI

- **链接**：
  - HF: https://huggingface.co/datasets/Metacreation/GigaMIDI
  - repo: https://github.com/Metacreation-Lab/GigaMIDI-Dataset
- **格式**：Large MIDI corpus with metadata, splits, loop info, expressive fields.
- **规模**：Paper / project: 1.43M+ unique MIDIs, 5.3M+ tracks, 1.8B notes. HF data card version seen here: 2,136,218 files and 6,891,738 tracks.
- **覆盖范围**：Broad all-genre corpus; classical, game, rock, and pop are especially visible; jazz exists but is not dominant.
- **Harmony 情况**：No unified chord labels.
- **样本结构**：{'split': 'train', 'md5': '...', 'music': b'MThd...', 'num_tracks': 4, 'loop_track_idx': [0, 2], 'loop_instrument_type': ['piano', 'drums'], 'music_styles_curated': ['classical']} 
- **评论**：Very good for scale and style diversity. Good source for piano/jazz subsetting and expressive modeling, but not direct chord supervision.

### 3.4 Discover MIDI Dataset

- **链接**：
  - HF: https://huggingface.co/datasets/projectlosangeles/Discover-MIDI-Dataset
- **格式**：MIDI + extensive precomputed metadata/features/search index.
- **规模**：6.74M+ unique, deduplicated, normalized MIDIs.
- **覆盖范围**：Massive MIR / discovery / symbolic-AI corpus with rich metadata and search tooling.
- **Harmony 情况**：No manual chord truth, but the dataset card says it provides a 321-chord harmonic vocabulary as part of the feature/signature system.
- **样本结构**：md5, path, genre labels, artist/title, karaoke/lyrics matches, monophonic melody summaries, pitches-patches counts, quality metrics.
- **评论**：Feels more like an engineering-friendly MIDI warehouse. Good for retrieval, statistics, and pretraining, not for direct chord labels.

### 3.5 Godzilla MIDI Dataset

- **链接**：
  - HF: https://huggingface.co/datasets/projectlosangeles/Godzilla-MIDI-Dataset
- **格式**：MIDI + metadata + search/filter index.
- **规模**：5.8M+ unique, de-duped, normalized MIDIs.
- **覆盖范围**：Large searchable corpus similar in spirit to Discover, with strong de-duplication and metadata collection.
- **Harmony 情况**：No manual chord truth.
- **样本结构**：md5 / metadata / basic signatures / text captions / feature counts.
- **评论**：Main value is scale and retrieval. Use it as background corpus or for pretraining and filtering.

### 3.6 Lakh MIDI Dataset (LMD)

- **链接**：
  - repo: https://github.com/craffel/midi-dataset
  - project: https://colinraffel.com/projects/lmd/
- **格式**：MIDI, with MSD audio matching / alignment information.
- **规模**：About 177K songs; different public pages show 176,581 or 178,561, so note the version when citing.
- **覆盖范围**：Large Western pop / rock / electronic corpus aligned to the Million Song Dataset.
- **Harmony 情况**：No unified harmony / chord annotation.
- **样本结构**：MIDI file + MSD audio match information.
- **评论**：Classic big corpus. Good for weak supervision and style diversity, but harmony must be extracted yourself.

### 3.7 MAESTRO

- **链接**：
  - dataset page: https://magenta.tensorflow.org/datasets/maestro
- **格式**：High-precision aligned audio + MIDI, with pedal/timing detail.
- **规模**：About 200 hours of professional piano performances (version dependent).
- **覆盖范围**：Classical solo piano, competition / concert level.
- **Harmony 情况**：No explicit chord labels.
- **样本结构**：composer / title / year / audio / MIDI / timing / pedal.
- **评论**：Best for human-like piano performance, timing, velocity, and pedal. Not a direct jazz-harmony source.

### 3.8 POP909

- **链接**：
  - repo: https://github.com/music-x-lab/POP909-Dataset
- **格式**：MIDI + beat / chord / key text files.
- **规模**：909 pop songs.
- **覆盖范围**：Pop-song arrangement generation; main melody, bridge, accompaniment piano, and multiple versions.
- **Harmony 情况**：Yes. `chord_audio/beat_audio.txt` stores chord names with start/end times.
- **样本结构**：index.mid, beat_audio/beat_midi.txt, chord_audio/beat_audio.txt, key_audio.txt, versions/.
- **评论**：Very practical for melody-to-chord / pop arranging. Not jazz-specific, but a strong harmony baseline.

### 3.9 Slakh2100

- **链接**：
  - repo: https://github.com/ethman/Slakh
- **格式**：Multitrack audio + aligned MIDI, rendered from MIDI with sampled instruments.
- **规模**：2100 automatically mixed tracks, about 145 hours, 187 patches, 34 classes.
- **覆盖范围**：Standardized multitrack instrumentation for separation, transcription, and multitrack generation.
- **Harmony 情况**：No unified chord labels, but the multitrack structure is excellent for voicing/orchestration learning.
- **样本结构**：mixture audio + aligned MIDI + track-level instrument metadata.
- **评论**：Useful for learning how to place harmony across instruments, not for direct chord supervision.

### 3.10 EMOPIA

- **链接**：
  - repo: https://github.com/annahung31/EMOPIA
  - emotion cls repo: https://github.com/SeungHeonDoh/EMOPIA_cls
- **格式**：Audio + MIDI + emotion labels.
- **规模**：387 songs, 1,087 clips in the original paper.
- **覆盖范围**：Pop piano covers/performances, including Japanese anime, K-pop, Western pop covers, movie soundtracks, and personal compositions.
- **Harmony 情况**：No unified chord labels in the original release.
- **样本结构**：{'emo_class': '1', 'YouTube_ID': '0vLPYiPN7qY', 'seg_id': '0'}
- **评论**：Best for emotion control and expressive piano modeling. Harmony must be extracted if you want to use it for arranger training.

### 3.11 EMOPIA+

- **链接**：
  - repo: https://github.com/Yuer867/EMO-Disentanger
  - data: https://zenodo.org/records/13122742
- **格式**：Functional representation / lead sheet / full performance eventization.
- **规模**：The paper uses 1,071 EMOPIA clips for finetuning; it also uses 18,206 HookTheory lead sheets for pretraining the lead-sheet stage.
- **覆盖范围**：Emotion-driven piano performance generation with a two-stage lead-sheet + performance setup.
- **Harmony 情况**：Yes. Melody and chords are encoded in a key-relative functional representation using Roman numerals.
- **样本结构**：lead sheet: melody + chord + key; full performance: melody + texture + bass.
- **评论**：This is one of the most relevant datasets for your use case because it ties harmony to performance directly.

### 3.12 HookTheory / HLSD

- **链接**：
  - Sheet Sage: https://github.com/chrisdonahue/sheetsage
  - data: https://sheetsage.s3.amazonaws.com/hooktheory/Hooktheory.json.gz
- **格式**：JSON.gz lead-sheet annotations.
- **规模**：Sheet Sage says 50 hours of aligned melody + harmony annotations; the dataset notebook reads about 26,175 entries.
- **覆盖范围**：Western / English pop lead sheets with aligned melody and harmony.
- **Harmony 情况**：Yes. The dataset includes MELODY and HARMONY tags plus keys, chords, and beats.
- **样本结构**：{"tags": ["AUDIO_AVAILABLE", "MELODY", "HARMONY"], "annotations": {"keys": [...], "chords": [...], "beats": [...]}}
- **评论**：Probably the single best source here for melody-to-chord learning.

### 3.13 MetaScore

- **链接**：
  - paper: https://arxiv.org/abs/2410.02084
- **格式**：Score + rich metadata + user tags + pseudo natural-language captions.
- **规模**：963K musical scores.
- **覆盖范围**：Forum-derived score data with rich text metadata, aimed at controllable generation.
- **Harmony 情况**：No unified manual chord truth.
- **样本结构**：score + user tags + rewritten pseudo captions + metadata.
- **评论**：Good if you want natural-language control, style descriptors, or caption-conditioned generation.

### 3.14 XMIDI

- **链接**：
  - No stable official source was verified in this session.
- **格式**：MIDI + emotion labels (public descriptions only).
- **规模**：Public descriptions I saw earlier mentioned about 108K MIDIs and 11 emotion classes, but I did not re-verify a canonical source here.
- **覆盖范围**：Emotion / style control.
- **Harmony 情况**：No explicit harmony labels.
- **样本结构**：MIDI + emotion label(s).
- **评论**：Treat as auxiliary emotion data unless you independently verify the canonical repository or paper.

### 3.15 Community anime MIDI datasets

- **链接**：
  - No unified official corpus; usually scattered across GitHub / Kaggle / personal dumps.
- **格式**：Usually MIDI, sometimes with simple title/label metadata.
- **规模**：No single citable official total.
- **覆盖范围**：J-Pop, anime OST, and Japanese-style piano arrangements.
- **Harmony 情况**：Unstable; depends on the specific repository.
- **样本结构**：Usually one MIDI file plus a title or tag.
- **评论**：Useful as a style supplement, but verify copyright and annotation quality carefully before using it for harmony supervision.

---

## 4. 如果你的目标是 jazz arranger，我的建议

### 第一层：先把 harmony 学出来
优先顺序：
1. **HookTheory / HLSD**：最直接的 melody-chord 对齐。
2. **POP909**：结构很清楚，适合 pop harmony。
3. **EMOPIA+**：适合把 lead sheet 和 performance 连接起来。
4. **PDMX**：做谱面级补充，尤其是公有领域可用性高。

### 第二层：再扩大语料和风格
5. **MetaMIDI / GigaMIDI / Discover / Godzilla / LMD**：做大规模预训练、风格迁移、jazz 子集筛选。
6. **MAESTRO / Slakh**：补 performance、voicing、编配感。
7. **MetaScore / XMIDI / anime MIDI**：做文本控制、情绪控制、风格补充。

### 第三层：训练策略建议
- 先训练一个 **chord-aware lead sheet encoder**。
- 再加一个 **voicing / arrangement decoder**。
- 最后把 jazz 语法拆成：
  - `melody -> chord progression`
  - `chord progression -> voicing`
  - `melody + voicing -> full arrangement`

如果你愿意，我下一步可以继续把这份文档整理成一个更像项目内规范说明的版本，或者再补一个“这些数据怎么拼成 jazz arranger 训练集”的小节。