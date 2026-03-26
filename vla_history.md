## 1. 前史: VLA はどうやって生まれたか

RT-2（2023.07）が「VLA（Vision-Language-Action model）」という用語を定義し、その最初の実装を示した。RT-2の設計は**2つの独立した系譜の合成**である：

1. **ロボット学習プロトコルとデータ（RT-1起点）** — 行動の離散化（256bin）、大規模デモデータ（13万エピソード）、評価プロトコル。RT-2はこれらをRT-1から継承した
2. **Web規模の事前学習済みVLM（PaLI-X / PaLM-E起点）** — RT-2は既存VLMを出発点に、行動をテキストトークンとして出力させることで、VLMのWeb知識を低レベルロボット制御に直接転用した

RT-2の核心的イノベーションは「行動のトークン化（action tokenization）」— ロボットの連続行動をVLMのトークナイザに載る文字列として表現し、VLMの自然言語出力と同じ枠組みで行動を生成させたこと。さらに、ロボットデモデータとWebのVLMデータ（VQA等）を混ぜるco-fine-tuningにより、汎化性能を大幅に向上させた。

### 1.1 ロボット学習のスケーリング: RT-1が作った土台

RT-2が継承した**ロボット側の系譜**。行動の離散化、大規模デモデータ収集、言語条件付きマルチタスク学習といったプロトコルは、以下の研究を経てRT-1で確立された。事前学習済みVLMをバックボーンとして使っていない。

※ 年はarXiv v1投稿日基準。

| arXiv投稿日 | 手法 | 著者・所属 | 役割 | 核心的貢献 |
|------------|------|-----------|------|-----------|
| 2021-06-02 | **Decision Transformer** | Chen et al. (UC Berkeley) | **系列モデリングの転換** | RLを「return-conditioned な系列生成」に再定式化（NeurIPS 2021）。評価はAtari/OpenAI Gym中心で、ロボット操作そのものが主題ではない |
| 2021-06-03 | **Trajectory Transformer** | Janner et al. (UC Berkeley) | **系列モデリング（計画寄り）** | DTの翌日に投稿。同じく軌跡の系列モデリングだが、ビームサーチで「計画」する発想が中核（NeurIPS 2021 spotlight）。RT-1が両者を"着想源"として明示的に引用 |
| 2021-09-24 | **CLIPort** | Shridhar et al. (UW/NVIDIA) | **VLM→ロボットの橋渡し** | CLIPの意味理解（what）+ TransporterNetの空間精度（where）を融合。10シミュレーション+9実機タスクのマルチタスクポリシー。**事前学習済みVLMの知識をロボット制御に転用する先駆**（CoRL 2021） |
| 2022-02-04 | BC-Z | Jang et al. (Google) | 言語条件付けBC | 100タスク・25,877エピソードでの言語条件付きゼロショット汎化。事前学習済みsentence encoder（USE系）で512次元埋め込み→FiLMでポリシーを条件づけ |
| 2022-03-23 | **R3M** | Nair et al. (NYU/Meta) | **視覚表現の事前学習** | Ego4D動画から時間対照学習+動画言語アライメントでロボット操作用の視覚表現を学習。RT-2が先行研究として明示的に引用・比較 |
| 2022-05-12 | Gato | Reed et al. (DeepMind) | マルチタスク統合 | 604タスク（制御系596+VL/言語8）を1つのTransformerで学習。ただし実機ロボットタスクはRGB Stacking（Sawyer）の1タスクのみ。「汎用エージェント」の概念を提示 |
| 2022-12-13 | **RT-1** | Brohan et al. (Google) | **スケーリングの実証** | **13ロボット・13万エピソード・744タスク**でTransformerを学習。スケーリングの有効性を実証。**VLAの直接的な土台** |

> **なぜRT-1はVLAではないのか？**
>
> RT-1のアーキテクチャは論文本体で以下のように明示されている：
>
> ```
> RT-1のアーキテクチャ（論文記載に基づく）:
>
>   画像(6枚) → ImageNet事前学習 EfficientNet-B3 ─┐
>                  ↑ FiLM層で言語条件づけ           │
>   言語指示 → Universal Sentence Encoder(512d) ───┘
>                                                    ↓
>                                    "vision-language tokens"
>                                    （言語で条件づけた視覚トークン）
>                                                    ↓
>                                              TokenLearner
>                                           （トークン数を圧縮）
>                                                    ↓
>                                     Decoder-only Transformer (8層)
>                                                    ↓
>                                           トークン化された行動
> ```
>
> 視覚はImageNet事前学習のEfficientNet-B3、言語はUSE埋め込みという**別系統の事前学習器をFiLMで結合**する設計。RT-1論文中で "vision-language tokens" という語を使うが、これは「言語で条件づけた視覚トークン」の意であり、Web規模で事前学習された統合VLMとは異なる。
>
> RT-2はRT-1の行動離散化プロトコルとデータを継承しつつ、モデル自体は既存VLM（PaLI-X 55B / PaLM-E 12B）を出発点とした。行動をVLMのテキストトークンとして表現し、ロボットデモとWebデータのco-fine-tuningでVLMのWeb知識がロボット制御に転移することを実証した。

**RT-2が継承したもの（引用関係で検証済み）**:

RT-1論文はDT/TTを**明示的に"着想源"と記述**しており、系列モデリングの発想がRT-1の設計に直接影響している。RT-2はRT-1から以下を継承した：
- **行動離散化プロトコル**: 各連続次元を256binに離散化してトークン列として出力する方式
- **デモデータセット**: RT-1由来の13万エピソード（モバイルマニピュレータ、7スキル種別）
- **評価プロトコル**: 同一環境での汎化評価。RT-2はRT-1の32%→62%（Unseen Average）と改善を報告

一方、CLIPort・R3MはRT-2論文で先行研究として引用されており、「事前学習済みモデルの知識はロボットに転用できる」ことを示した前例として位置づけられている。

---

### 1.2 VLMの成熟: RT-2のモデル側の系譜

RT-2が**モデルの出発点**としたのは、PaLI-X（55B）とPaLM-E（12B）という2つの既存VLMである。RT-2は「新しいパラメータを追加せずに、既存VLMから行動を出力させる」と明記しており、VLMの成熟がVLA実現の前提条件だった。

特にPaLM-E（2023.03、RT-2より4か月先行）は、連続センサ入力をLLMのトークン空間に注入する「Embodied MLLM」として、RT-2の概念的先行者でもある。RT-2はPaLM-Eの「multimodal sentences」という枠組みを参照しつつ、出力を高レベル計画ではなく低レベル行動に拡張した。

VLAの性能は (1) Vision Encoderの視覚表現の質、(2) LLMの表現力、(3) 両者の接続方法 に強く依存する。

#### Vision Encoderの系譜

VLAが使うVision Encoderは単純なViTではなく、学習方法の違いで複数の系統がある。

| 年 | モデル | 著者・所属 | 学習方法 | 核心的貢献 |
|----|--------|-----------|---------|-----------|
| 2020 | **ViT** | Dosovitskiy et al. (Google Brain) | 教師あり分類 | 画像をパッチ→トークンでTransformer処理。視覚表現のパラダイムシフト（arXiv: 2020-10、ICLR 2021採択） |
| 2021 | **CLIP ViT** | Radford et al. (OpenAI) | 対照学習（softmax） | 4億画像-テキストペアで対照学習。視覚-言語共通埋め込みの確立（arXiv: 2021-02-26） |
| 2023 | **DINOv2** | Oquab et al. (Meta) | **自己教師あり学習** | 言語ラベルなしで画像だけから空間的・局所的な視覚特徴を学習。**低レベルの空間情報（把持点・物体境界等）に強い** |
| 2023 | **SigLIP** | Zhai et al. (Google) | 対照学習（**sigmoid**） | CLIPのsoftmax正規化をpairwise sigmoid lossに変更。バッチサイズに依存しにくく、fine-tuneしても視覚表現が崩れにくい（arXiv: 2023-03-27） |
| 2024 | **SigLIP2** | Google | 対照学習 + 自己教師あり + captioning | SigLIPに自己教師ありloss・captioningベース事前学習・オンラインデータキュレーションを統合。汎用性がさらに向上 |

**VLAでSigLIPが主流になりつつある理由**:
- PaLI-3がSigLIP事前学習ViTと分類事前学習ViTを比較し、マルチモーダル（特にlocalizationや視覚的テキスト理解）でSigLIPが優位と報告
- PaliGemma / PaliGemma 2がSigLIP-So400mを標準採用 → π0, SpatialVLAに直結
- fine-tune時にsoftmax正規化（CLIPの弱点）が不要なため、ロボットデータでの追加学習で表現が崩れにくい

**OpenVLAがSigLIPとDINOv2を併用する理由**:
- SigLIP = **高レベルの意味情報**（「これはりんごだ」）
- DINOv2 = **低レベルの空間情報**（「ここに掴める面がある」）
- Prismatic VLMsの研究で、この2つのfused backbonesが設計全体の性能を押し上げることを確認

#### VLMの系譜（VLAのバックボーン）

VLMには「Vision EncoderとLLMをどう結ぶか」で複数のアーキテクチャパターンがある。

##### アーキテクチャパターン

| パターン | 代表 | 構造 | VLAとの相性 |
|---------|------|------|-----------|
| Dual Encoder型 | CLIP | 画像とテキストを別々にエンコード→共通空間 | △ 生成タスクに弱い |
| Encoder-Decoder型 | PaLI系 | Vision Encoder + テキストDecoder | ○ 生成可能だがやや古い |
| **Decoder-only + Visual Tokens型** | LLaVA, PaliGemma | Vision EncoderのトークンをLLMに入力 | **◎ VLAで主流** |
| Q-Former接続型 | BLIP-2 | 学習可能なクエリで視覚情報を圧縮→LLMへ | △ 圧縮で情報が落ちうる |
| 統合VLM型 | Qwen-VL | Vision Encoder + LLMを最初から統合設計 | ◎ 設計の自由度が高い |

**VLAでDecoder-only型が有利な理由**:
1. RT-2が示したように、行動をテキストトークンと同形式にすると学習目標が「次トークン予測」に自然に落ちる → Decoder-only LMの学習形式と整合
2. VLM側（Llama 2, Gemma等）のスケールや指示追従能力をそのまま制御に転用しやすい
3. VLMを「特徴抽出器」としても「生成器」としても扱える柔軟性がある

##### PaLI系の系譜（Google → VLAの主要バックボーン）

```
PaLI (2022.09)          PaLI-X (2023.05)         PaLI-3 (2023)
ViT-e(4B) + mT5    →   ViT-22B + 大規模化    →  SigLIP ViTが優位と実証
                         ↓ RT-2のバックボーン
                                                      ↓
                                                 PaliGemma (2024.07)
                                                 SigLIP-So400m + Gemma-2B
                                                 軽量・オープン・転移重視
                                                 ↓ π0 / π0-FASTのバックボーン
                                                      ↓
                                                 PaliGemma 2 (2024.12)
                                                 SigLIP-So400m + Gemma 2 (2B〜27B)
                                                 複数解像度(224/448/896)で段階学習
                                                 ↓ SpatialVLAのバックボーン
```

**要点**: 「PaLI/PaLI-X（巨大・専用）→ PaliGemma（小型・オープン・転移重視）→ PaliGemma 2（多様なサイズ&解像度）」という、実運用・研究展開のしやすさに寄った分岐。

##### その他のVLA関連VLM

| 年 | モデル | 著者・所属 | 核心的貢献 | VLAとの関係 |
|----|--------|-----------|-----------|-----------|
| 2022 | Flamingo | Alayrac et al. (DeepMind) | 凍結VisionEncoder + LLMをPerceiver Resamplerで接続。Few-shot VLMのテンプレート確立 | アーキテクチャパターンの先駆 |
| 2023 | **PaLM-E** | Driess et al. (Google) | 連続センサ入力をLLM入力に混在させるEmbodied MLLM。**RT-2のもう一つのバックボーン** | VLMだけでなくEmbodied MLLMもVLAの背骨になりうることを実証 |
| 2023 | **LLaVA** | Liu et al. (UW/Microsoft) | CLIP ViT + Vicuna を線形射影で接続。シンプルなアーキテクチャでGPT-4Vに迫る性能 | Decoder-only + Visual Tokens型の代表。VLA設計の参照点 |
| 2023 | BLIP-2 | Li et al. (Salesforce) | Q-Formerで凍結エンコーダとLLMを効率的に橋渡し | Q-Former型の代表 |
| 2024 | **Prismatic VLMs** | Karamcheti et al. (ICML 2024) | VLMの設計軸を系統評価。SigLIP + DINOv2のfused backbonesが有効と実証 | **OpenVLAのバックボーン（Prismatic-7B）** |
| 2023– | **Qwen-VL系** | Alibaba | 動的解像度・マルチモーダルRoPE等を導入したVLMシリーズ（Qwen-VL → Qwen2-VL → Qwen3-VL） | **VLANeXtの最終設定がQwen3-VL 2Bを採用**。CoA-VLAのDi-VLAもQwen2-VLを統合 |
| 2023– | **InternVL** | Shanghai AI Lab | 6B規模のvision foundation modelをスケール。LLMと整合 | VLA-Critic（InternVL上に報酬/批評を統合）で利用 |



#### LLMのロボット応用（VLAとは異なるアプローチ）

VLMをバックボーンに使うのではなく、LLMを「ロボットの知恵袋」として外部利用する試み。VLAとは別系統だが、「言語モデルの知識がロボットに有用である」ことを示した重要な前例。

| 年 | 手法 | 著者・所属 | 核心的貢献 |
|----|------|-----------|-----------|
| 2022 | **SayCan** | Ahn et al. (Google) | LLMがプラン提案、value functionが実行可能性をスコアリング。**LLM知識のロボットへの接地（grounding）**。ただし行動生成は別のポリシーが担当 |
| 2023 | Code as Policies | Liang et al. (Google) | LLMにロボット制御コード（Python）を直接生成させる。行動値ではなくコードを出力 |

#### まとめ: 2つの系譜がRT-2で合成された

2023年までにVLMは「Vision Encoderの高品質化（SigLIP）」「Decoder-onlyアーキテクチャの確立」「PaLI系のスケーリング」「PaLM-Eによる身体性タスクへの接続」を通じて十分に成熟した。RT-2はこの成熟したVLM（PaLI-X / PaLM-E）をモデルの出発点とし、RT-1が確立したロボット学習プロトコル（行動離散化・大規模デモデータ）を接続することで、VLAを実現した。SayCan/Code as Policiesは「LLMの知識はロボットに役立つ」という確信を与えた前例。

### VLAの誕生図

```
  ■ ロボット学習プロトコル/データ          ■ Web規模VLM
  ──────────────────────────            ──────────────
  DT/TT (2021)                          ViT (2020)
  「Transformerで行動系列を生成」         CLIP (2021) → SigLIP (2023)
          │ ※RT-1が"着想源"と明記        PaLI (2022) → PaLI-X (2023)
          ↓                                    │
  ┌─────────────────────┐              PaLM-E (2023.03)
  │  RT-1 (2022.12)     │              Embodied MLLM
  │  行動離散化(256bin)  │              「連続センサ入力をLLMへ注入」
  │  13万ep / 744タスク  │              "multimodal sentences"
  │  ※VLAではない        │                    │
  └────────┬────────────┘                    │
           │                                  │
           │  プロトコル/データ                │  モデル/重み
           │  を継承                           │  を出発点に
           │                                  │
           └──────────┬───────────────────────┘
                      │
                      ↓  action tokenization
  ┌──────────────────────────────────────────────┐
  │           RT-2 (2023.07) = VLA 誕生           │
  │                                                │
  │  ・行動をVLMのテキストトークンとして表現        │
  │  ・ロボットデモ + Web VLMデータのco-fine-tune  │
  │  ・「VLA」というカテゴリを定義                  │
  │  ・バックボーン: PaLI-X 55B / PaLM-E 12B      │
  │  ・汎化: RT-1の32% → RT-2の62% (Unseen Avg)   │
  └──────────────────────────────────────────────┘
```

**読み方**: RT-2は「RT-1の改良版」ではなく、**ロボット学習プロトコル/データ（RT-1起点）** と **Web規模VLM（PaLI-X/PaLM-E起点）** の合成として生まれた。RT-2の核心的イノベーションは、両者を繋ぐ「action tokenization」— 行動をVLMのトークン列として表現し、同一の枠組みで学習・生成させたこと。

---
