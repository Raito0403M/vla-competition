# VLA コンペ運営者のための体系的知識ベース

> VLA（Vision-Language-Action）の全体像を、歴史的変遷から現在の課題まで整理する

---

## 目次

1. [前史: VLA以前の3つの流れ](#1-前史-vla以前の3つの流れ)
2. [VLAモデルの系譜](#2-vlaモデルの系譜)
3. [行動表現の技術的比較](#3-行動表現の技術的比較)
4. [重要データセット](#4-重要データセット)
5. [ベンチマーク一覧](#5-ベンチマーク一覧)
6. [触っておくべきライブラリ](#6-触っておくべきライブラリ)
7. [現在の課題（未解決問題）](#7-現在の課題未解決問題)
8. [2025年の注目トレンド](#8-2025年の注目トレンド)
9. [コンペ運営者として押さえるべきポイント](#9-コンペ運営者として押さえるべきポイント)

---

## 1. 前史: VLA以前の3つの流れ

VLAは3つの独立した研究の流れが2023年に合流して生まれた。

### 1.1 模倣学習（Imitation Learning）

| 年 | 手法 | 著者・所属 | 核心的貢献 |
|----|------|-----------|-----------|
| 1989 | ALVINN | Pomerleau (CMU) | ニューラルネットによる模倣の起源 |
| 2011 | **DAgger** | Ross, Gordon, Bagnell (CMU) | BCの分布シフト問題に対する理論的解法。ポリシー実行状態でエキスパートにラベル付けしデータを逐次拡張 |
| 2016 | **GAIL** | Ho & Ermon (Stanford) | GANフレームワークを模倣学習に適用。報酬関数の設計不要に |
| 2018 | RoboTurk | Mandlekar et al. (Stanford) | クラウドソーシング型大規模遠隔操作データ収集 |
| 2021 | **Robomimic** | Mandlekar et al. (Stanford/NVIDIA) | 模倣学習手法の体系的ベンチマーク。BC-RNNの有効性を実証 |

**VLAへの接続**: BCの枠組みは維持されつつ、Transformerとスケーリングで分布シフトをデータ量と表現力で緩和する方向へ進化。

### 1.2 ロボット基盤モデル

| 年 | モデル | 著者・所属 | 核心的貢献 |
|----|--------|-----------|-----------|
| 2021 | Decision Transformer | Chen et al. (UC Berkeley) | RLをシーケンスモデリング問題に再定式化 |
| 2022 | **RT-1** | Brohan et al. (Google) | **大規模実ロボットデータ（13万エピソード）でTransformerを学習**。スケールの有効性を実証。VLAの直接的先行研究 |
| 2022 | BC-Z | Jang et al. (Google) | 100+タスクでの言語条件付きゼロショット汎化BC |
| 2022 | CLIPort | Shridhar et al. (UW/NVIDIA) | CLIPの意味理解 + TransporterNetの空間精度を融合。**事前学習済みVLMのロボティクス転用**の先駆 |
| 2022 | Gato | Reed et al. (DeepMind) | 604タスクを1つのTransformerで学習。「汎用エージェント」の概念を提示 |
| 2023 | PerAct | Shridhar et al. (UW/NVIDIA) | 3Dボクセル上の言語条件付き6-DoF操作 |

**VLAへの接続**: 大規模データセット + Transformerアーキテクチャの有効性が、VLAの学習基盤に。

### 1.3 Vision-Language モデル（VLM）

| 年 | モデル | 著者・所属 | 核心的貢献 |
|----|--------|-----------|-----------|
| 2021 | **ViT** | Dosovitskiy et al. (Google Brain) | 画像をパッチ→トークンでTransformer処理。視覚表現のパラダイムシフト |
| 2021 | **CLIP** | Radford et al. (OpenAI) | 4億画像-テキストペアで対照学習。視覚-言語共通埋め込みの確立 |
| 2022 | Flamingo | Alayrac et al. (DeepMind) | 凍結VisionEncoder + LLMをPerceiver Resamplerで接続。Few-shot VLMのテンプレート確立 |
| 2022 | **SayCan** | Ahn et al. (Google) | **LLM知識のロボットへの接地（grounding）**。LLMがプラン提案、value functionが実行可能性をスコアリング |
| 2023 | Code as Policies | Liang et al. (Google) | LLMにロボット制御コード（Python）を直接生成させる |
| 2023 | **LLaVA** | Liu et al. (UW/Microsoft) | CLIP ViT + Vicuna を線形射影で接続。**シンプルなアーキテクチャでGPT-4Vに迫る性能**。オープンソースVLMの代表格 |
| 2023 | PaLI / PaLI-X | Chen et al. (Google) | ViT-e(4B) + mT5。大規模VLMスケーリングの実証 |
| 2023 | BLIP-2 | Li et al. (Salesforce) | Q-Formerで凍結エンコーダとLLMを効率的に橋渡し |

**VLAへの接続**: CLIP/ViTの視覚表現、LLMの推論能力、VLM（LLaVA, PaLI）のマルチモーダル統合がVLAの「バックボーン」に直接転用された。

### 3つの流れの合流図

```
模倣学習                    ロボット基盤モデル              Vision-Language
(BC, DAgger, GAIL)        (RT-1, BC-Z, CLIPort)         (CLIP, ViT, LLaVA, PaLI)
     │                          │                              │
     │    大規模BC               │    Transformer              │    事前学習済み
     │    + 言語条件付け          │    + 大規模データ             │    VLMバックボーン
     │                          │                              │
     └──────────────────────────┼──────────────────────────────┘
                                │
                          ┌─────┴─────┐
                          │  VLA 誕生  │
                          │  (RT-2, 2023)│
                          └───────────┘
                 事前学習済みVLMにロボット行動出力を
                 追加でfine-tuningするパラダイム
```

---

## 2. VLAモデルの系譜

### 2.1 RT-2（2023）— VLAの概念を確立

| 項目 | 内容 |
|------|------|
| **論文** | RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control |
| **著者** | Brohan et al. (Google DeepMind) |
| **arXiv** | [2307.15818](https://arxiv.org/abs/2307.15818) |
| **アーキテクチャ** | PaLI-X (55B) または PaLM-E (12B) をバックボーンに使用。画像+言語指示→行動トークンを出力 |
| **行動表現** | **離散トークン化**: 連続行動値を256ビンに離散化し、テキストトークンとして出力。例: "1 128 91 241 5 101 127" |
| **学習データ** | Web上のVLデータ + ロボットデータで共同学習。ロボットデータはRT-1と同じ13万エピソード |
| **公開** | **非公開**（Google内部モデル） |
| **核心的貢献** | **VLMのWeb知識がロボット制御に転移する**ことを初めて実証。「VLA」という用語を定義 |
| **限界** | 55Bモデルの推論が遅い（1-3Hz）。非公開で再現不可。離散化による行動精度の損失 |

### 2.2 RT-2-X（2023）— マルチロボットVLA

| 項目 | 内容 |
|------|------|
| **論文** | Open X-Embodiment Collaboration |
| **著者** | Google DeepMind + 21機関 |
| **核心的貢献** | Open X-Embodimentデータセット（22ロボット）でRT-2を学習。異なるロボット間の正の転移を実証 |
| **公開** | **非公開**（データセットのみ公開） |

### 2.3 Octo（2024）— 最初のオープンソースVLA

| 項目 | 内容 |
|------|------|
| **論文** | Octo: An Open-Source Generalist Robot Policy |
| **著者** | Ghosh, Walke et al. (UC Berkeley) |
| **arXiv** | [2405.12213](https://arxiv.org/abs/2405.12213) |
| **アーキテクチャ** | Transformerベース。画像（複数視点対応）+ 言語 + 独自体（embodiment）情報を入力。Diffusion Headで行動生成 |
| **行動表現** | **Diffusion Head**: ノイズから連続行動を反復的にデノイズ。マルチモーダル行動分布を表現可能 |
| **学習データ** | Open X-Embodimentの800Kエピソード（25ロボット混合） |
| **パラメータ** | 93Mの小型モデル |
| **公開** | **完全オープンソース**: コード + 重み + データ |
| **核心的貢献** | **最初の完全オープンソースVLA**。fine-tuneして新ロボットに適応可能な設計 |
| **限界** | 93Mと小型のため、言語理解・視覚理解がVLMベースのVLAに劣る。VLMバックボーンを使っていない |

### 2.4 OpenVLA（2024）— 7B VLAのオープンソース化

| 項目 | 内容 |
|------|------|
| **論文** | OpenVLA: An Open-Source Vision-Language-Action Model |
| **著者** | Kim, Khazatsky et al. (Stanford / UC Berkeley) |
| **arXiv** | [2406.09246](https://arxiv.org/abs/2406.09246) |
| **アーキテクチャ** | **Prismatic VLM**（SigLIP + DinoV2 視覚エンコーダ + Llama 2 7B LLM）をベースに、行動出力を追加学習 |
| **行動表現** | **離散トークン化**: RT-2と同様、連続行動を256ビンに離散化してテキストトークンとして自己回帰生成 |
| **学習データ** | Open X-Embodiment（970Kエピソード） |
| **パラメータ** | 7B |
| **公開** | **完全オープンソース**: コード + 重み。Hugging Faceで配布 |
| **核心的貢献** | **RT-2のオープンソース再現**。7Bスケールで誰でもVLAを試せる環境を実現。fine-tuneレシピも公開 |
| **限界** | 離散トークン化の精度限界。単一画像入力（複数視点非対応）。**LIBERO-PROで丸暗記が露呈** |

### 2.5 π0 / π0-FAST（2024）— Flow Matchingベースの行動生成

| 項目 | 内容 |
|------|------|
| **論文** | π0: A Vision-Language-Action Flow Model for General Robot Control |
| **著者** | Black, Nakamoto et al. (Physical Intelligence) |
| **arXiv** | [2410.24164](https://arxiv.org/abs/2410.24164) |
| **アーキテクチャ** | PaLI-Gemma (3B VLM) をバックボーンに使用。VLMの出力に**専用のAction Expert（別のTransformerブロック群）**を追加し、行動チャンクを生成 |
| **行動表現** | **Flow Matching**: ガウスノイズから連続行動チャンク（複数ステップ分の行動）へのフローを学習。Diffusionより学習安定・推論高速 |
| **学習データ** | 自社収集の大規模実ロボットデータ + Open X-Embodiment。7つの異なるロボットプラットフォーム |
| **パラメータ** | 3B（VLMバックボーン） + Action Expert |
| **公開** | **openpi**として PyTorch実装 + 重みを公開（2025年） |
| **核心的貢献** | **Flow Matchingで連続行動を高精度に生成**。離散化の精度損失を回避。行動チャンク（50ステップ先まで一度に予測）で長期行動を効率的に生成 |

**π0-FAST**:
| 項目 | 内容 |
|------|------|
| **行動表現の変更** | Flow Matchingの代わりに**FAST（Frequency-domain Action Sequence Tokenization）**を採用。行動チャンクをDCT（離散コサイン変換）で周波数領域に変換→トークン化 |
| **利点** | 純粋な自己回帰生成になるため、VLMの推論パイプラインをそのまま流用可能。推論効率が向上 |
| **核心的貢献** | 「連続行動のトークン化」の新しいアプローチ。RT-2の素朴な256ビン離散化より高精度 |

### 2.6 π0.5（2025）— Webデータとの大規模統合

| 項目 | 内容 |
|------|------|
| **論文** | π0.5: a Vision-Language-Action Model with Open-World Generalization |
| **著者** | Physical Intelligence |
| **核心的貢献** | Webデータ（テキスト、画像、ビデオ）とロボットデータを大規模に統合学習。片付け・料理など長期タスクに対応 |
| **公開** | openpiで一部公開。LIBERO用fine-tuneチェックポイントあり |
| **限界** | LIBERO-PROで位置摂動0.4ユニットで崩壊（ただしπ0よりは堅牢: 0.38 vs 0.00） |

### 2.7 HPT（2024）— 異種ロボットの統合

| 項目 | 内容 |
|------|------|
| **論文** | Heterogeneous Pre-trained Transformers |
| **著者** | Wang et al. (CMU) |
| **アーキテクチャ** | **異種ロボットの異なるセンサ・アクチュエータ**を統一フレームワークで扱う。ロボット固有のステム（入出力変換層）+ 共有Transformerトランク |
| **核心的貢献** | 異なるembodiment間でTransformerの内部表現を共有。52データソースから事前学習 |
| **公開** | コード + 重み公開 |

### 2.8 GR-2（2024）— ビデオ生成とVLAの融合

| 項目 | 内容 |
|------|------|
| **論文** | GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge |
| **著者** | ByteDance |
| **アーキテクチャ** | ビデオ生成モデル（次フレーム予測）を事前学習し、そこからロボット行動を生成。World Modelの発想 |
| **核心的貢献** | Webビデオで大規模事前学習 → 実ロボットデータで行動生成にfine-tune。**ビデオ理解を中間表現**として活用 |

### 2.9 SpatialVLA（2025）— 空間理解の強化

| 項目 | 内容 |
|------|------|
| **論文** | SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model |
| **著者** | RSS 2025 accepted |
| **arXiv** | [2501.15830](https://arxiv.org/abs/2501.15830) |
| **アーキテクチャ** | PaliGemma2バックボーン + **Ego3D Position Encoding**（深度推定で2D→3D位置情報を注入）+ **Adaptive Action Grids**（適応的離散化グリッド） |
| **学習データ** | **110万実ロボットエピソード** |
| **公開** | [GitHub](https://github.com/SpatialVLA/SpatialVLA) + Hugging Faceで重み公開 |
| **核心的貢献** | VLAに3D空間理解を組み込み。位置摂動に対するロバスト性の向上を目指す |

### 2.10 VLANeXt（2025）— VLA設計のレシピ集

| 項目 | 内容 |
|------|------|
| **論文** | VLANeXt: Recipes for Building Strong VLA Models |
| **arXiv** | [2602.18532](https://arxiv.org/abs/2602.18532) |
| **核心的貢献** | **12の設計知見**を体系化。基盤コンポーネント（LLaMA + SigLIP2）、知覚、行動モデリングの3軸で最適な選択肢を実験的に特定。**LIBERO / LIBERO-Plusの両方でSOTA** |
| **意義** | コンペ運営者にとって、「どの設計選択が性能に効くか」の最新エビデンス |

### その他の注目VLA（2025）

| モデル | 特徴 |
|--------|------|
| **VLA-0** | VLMの改造ゼロで行動出力を追加する手法 |
| **SimpleVLA-RL** | VLA + RL のスケーリング |
| **OpenVLA-OFT** | OpenVLAのfine-tune効率改善 |

---

### VLAモデル系譜の全体図

```
2023.07  RT-2 ──── VLAの概念確立（非公開、55B、離散トークン）
           │
2023.10  RT-2-X ── マルチロボット転移（非公開）
           │
           ├─────────────────────────────────────────────┐
           │                                             │
2024.05  Octo ──── 最初のOSS VLA                   2024.06  OpenVLA ── RT-2のOSS再現
         (93M, Diffusion Head)                           (7B, 離散トークン)
           │                                             │
2024.10  π0 ────── Flow Matching導入               2024    HPT ──── 異種ロボット統合
           │       (3B + Action Expert)                   │
2024.12  π0-FAST ─ FAST行動トークン化              2024    GR-2 ─── ビデオ生成→行動
           │                                             │
2025.01  π0.5 ──── Web+ロボットデータ統合          2025    SpatialVLA ─ 3D空間理解
           │                                             │
2025.02  ────────── VLANeXt: 設計レシピの体系化 ──────────┘
```

---

## 3. 行動表現の技術的比較

VLAの核心的な設計選択。コンペで参加者が直面する技術的トレードオフ。

### 3.1 離散トークン化（RT-2, OpenVLA）

```
連続行動 [0.12, -0.34, 0.78, ...] → 256ビンに離散化 → テキストトークン "1 128 91"
```

| 利点 | 欠点 |
|------|------|
| VLMの自己回帰生成をそのまま利用可能 | 256ビンの精度限界（ロボット制御には粗い） |
| 実装がシンプル | マルチモーダル行動分布を表現できない |
| VLMの推論最適化をそのまま適用可 | 各次元を独立に離散化→次元間相関を失う |

### 3.2 Diffusion Policy（Octo）

```
ガウスノイズ z ~ N(0,I) → 反復デノイズ → 連続行動 a
```

| 利点 | 欠点 |
|------|------|
| マルチモーダル行動分布を表現可能 | デノイズの反復が必要で推論が遅い |
| 連続行動を高精度に生成 | ハイパーパラメータ（ノイズスケジュール等）の調整が多い |
| 学術的に十分検証された手法 | VLMパイプラインとの統合が非自明 |

### 3.3 Flow Matching（π0）

```
ガウスノイズ z ~ N(0,I) → 決定的ベクトル場で輸送 → 連続行動 a
```

| 利点 | 欠点 |
|------|------|
| Diffusionより**学習が安定**、ハイパーパラメータが少ない | VLMとの統合に専用Action Expertが必要 |
| **推論が高速**（より少ないステップ数で収束） | 学習時間はAutoregressive より遅い |
| 行動チャンク（50ステップ分同時予測）が自然 | 比較的新しく、ベストプラクティスが固まりきっていない |
| 連続値のためロボット制御精度が高い | |

### 3.4 FAST（π0-FAST）

```
連続行動チャンク → DCT（周波数変換）→ トークン化 → 自己回帰生成
```

| 利点 | 欠点 |
|------|------|
| VLMのAutoregressive生成をそのまま流用 | DCTの圧縮で情報損失の可能性 |
| 推論効率がFlow Matchingより高い場合がある | 高周波成分（細かい動き）の表現に限界 |
| 素朴な256ビン離散化より高精度 | 比較的新しい手法 |

### 比較まとめ

| 手法 | 精度 | 推論速度 | マルチモーダル | VLM統合の容易さ | 代表モデル |
|------|------|---------|--------------|----------------|-----------|
| 離散トークン | △ | ○ | × | ◎ | RT-2, OpenVLA |
| Diffusion | ◎ | △ | ◎ | △ | Octo |
| Flow Matching | ◎ | ○ | ○ | △ | π0 |
| FAST | ○ | ◎ | △ | ◎ | π0-FAST |

---

## 4. 重要データセット

### 実ロボットデータセット

| データセット | 年 | 規模 | ロボット | 特徴 | URL |
|-------------|-----|------|---------|------|-----|
| **Open X-Embodiment** | 2023 | 100万+エピソード | 22種類のロボット | 21機関のデータを統合。最大規模。RT-2-Xの学習に使用 | [公式](https://robotics-transformer-x.github.io/) |
| **Bridge Data V2** | 2023 | 60,096軌道 | WidowX | 24環境、13ロボット。UCBerkeley。オープンソースの代表格 | [公式](https://rail-berkeley.github.io/bridgedata/) |
| **DROID** | 2024 | 76,000デモ | Franka Emika Panda | 分散収集（複数機関）。350シーン、多様な操作タスク | [公式](https://droid-dataset.github.io/) |
| **RoboSet** | 2023 | 100,000+軌道 | Franka | マルチスキル（pick, place, wipeなど）。MIT | [公式](https://robopen.github.io/) |
| **RoboNet** | 2019 | 15,000+時間 | 7ロボット | 複数機関からのビデオデータ収集の先駆け | UC Berkeley |

### コンペ運営者として知っておくべきこと
- Open X-Embodiment は**データ品質が不均一**（機関によって収集方法・品質がバラバラ）
- Bridge Data V2 は最も広く使われている研究用オープンデータ
- DROID は品質管理が比較的しっかりしている分散収集の成功例
- **ロボットデータの規模はWeb画像データの1/10,000以下**（VLMの学習データは数十億、ロボットは数十万）

---

## 5. ベンチマーク一覧

### シミュレーションベンチマーク

| ベンチマーク | 年 | シミュレータ | タスク数 | 特徴 | コンペ利用適性 |
|------------|-----|-----------|---------|------|-------------|
| **LIBERO** | 2023 (NeurIPS) | MuJoCo/robosuite | 4スイート×10タスク | 汎化の軸を分離評価（Spatial/Object/Goal/Long） | **◎ コンペのベース** |
| **LIBERO-PRO** | 2025 | LIBERO拡張 | +16スイート | 4つの摂動軸で丸暗記問題を暴く | **◎ Phase 2で利用** |
| **LIBERO-Plus** | 2025 | LIBERO拡張 | 7次元摂動 | カメラ・照明・背景・ロボット初期状態なども追加 | ○ 参考 |
| **SIMPLER** | 2024 | MuJoCo/Sapien | RT-1/RT-2タスク | 実機モデルをシミュレーションで近似評価 | ○ |
| **RLBench** | 2020 | CoppeliaSim | 100タスク | 3D操作の大規模ベンチマーク。Imperial College London | △ やや古い |
| **MetaWorld** | 2020 | MuJoCo | 50タスク | メタ学習・マルチタスク学習向け。Sawyer環境 | △ |
| **ManiSkill** | 2023 | SAPIEN | 多数 | GPU並列化で高速評価。UCSD/Hillbot | ○ |
| **CALVIN** | 2022 | PyBullet | 長期操作 | 連続マルチタスク。言語条件付き。Freiburg大 | ○ |

### LIBEROの4スイート詳細（コンペのベース）

| スイート | 汎化軸 | 内容 | 難易度 |
|---------|--------|------|--------|
| **LIBERO-Spatial** | 空間配置 | 同じタスク、物体の位置が異なる | ★★☆ |
| **LIBERO-Object** | 物体種類 | 同じ空間、異なる物体 | ★★☆ |
| **LIBERO-Goal** | ゴール（言語理解） | 同じシーン、異なるゴール指示 | ★★★ |
| **LIBERO-Long** | 長期系列 | 10ステップの連続タスク | ★★★★ |

各スイート: 10タスク × 50デモ、BDDLで手続き的に生成、評価は500試行のsuccess rate

---

## 6. 触っておくべきライブラリ

### 必須（コンペインフラとして使用）

| ライブラリ | 提供元 | 用途 | 優先度 |
|-----------|--------|------|--------|
| **openpi** | Physical Intelligence | π0/π0-FASTの公式実装。fine-tune + 推論 + LIBERO評価の一気通貫パイプライン | ★★★ 最優先 |
| **LIBERO** | UT Austin | ベンチマーク環境。robosuite上のタスク定義 | ★★★ 最優先 |
| **LIBERO-PRO** | HUST | 摂動評価の拡張。evaluation_config.yaml + perturbation.py | ★★★ Phase 2用 |
| **robosuite** | Stanford | MuJoCoベースのロボットシミュレーション基盤 | ★★☆ LIBEROの依存 |

### 推奨（知識として理解、可能なら触る）

| ライブラリ | 提供元 | 用途 | 優先度 |
|-----------|--------|------|--------|
| **LeRobot** | Hugging Face | ロボット学習の統一フレームワーク。データ管理・可視化・学習パイプライン。**openpiとのデータ連携あり** | ★★☆ |
| **robomimic** | Stanford/NVIDIA | 模倣学習のベンチマーク。BC/BC-RNN等の実装。「模倣学習とは何か」を学ぶのに最適 | ★★☆ 教育的 |
| **MuJoCo** | DeepMind (OSS) | 物理シミュレータ。robosuite/LIBEROの基盤 | ★☆☆ 依存関係 |
| **OpenVLA** | Stanford/Berkeley | 7B VLAの参照実装。RT-2アーキテクチャを理解するのに有用 | ★★☆ 比較用 |

### openpiの構成（最も重要）

```
openpi/
├── scripts/
│   ├── train.py              # 学習スクリプト
│   ├── serve_policy.py       # 推論サーバー
│   └── compute_norm_stats.py # 正規化統計計算
├── examples/
│   └── libero/
│       ├── Dockerfile        # Docker化された評価環境
│       ├── compose.yml       # Docker Compose設定
│       ├── main.py           # 評価スクリプト
│       └── convert_libero_data_to_lerobot.py  # データ変換
├── third_party/
│   └── libero/               # LIBEROのsubmodule
└── configs/
    ├── pi0_libero             # π0のLIBERO用config
    ├── pi0_fast_libero        # π0-FASTのLIBERO用config
    └── pi05_libero            # π0.5のLIBERO用config
```

---

## 7. 現在の課題（未解決問題）

### 7.1 汎化の課題 — 最重要

**LIBERO-PROが暴いた「丸暗記」問題**:
- 標準LIBEROで90%超 → 物体位置0.2ユニットずらすだけで0%に崩壊
- 指示を無意味文字列「fdsgfdsgsd」にしても同じ軌道を出力 → **言語を一切読んでいない**
- 対象物体を除去しても同じ軌道 → **視覚知覚もしていない**
- サブタスク合成（A→B）で完全失敗 → **因果理解がない**

**Sim2Real Gap**:
- シミュレーションの画像品質・物理特性が実世界と異なる
- SIMPLERが近似評価を試みているが、完全な解決には至っていない

**Compositional Generalization**:
- 個々のスキルは学習できるが、組み合わせると崩壊する
- VLAは「スキルの辞書」を持っているのではなく、「軌道の暗記帳」に近い状態

### 7.2 スケーリングの課題

**データの希少性**:
```
VLMの学習データ:  ~数十億 画像-テキストペア
ロボットデータ:    ~数百万 エピソード（全世界合計）
比率:             1 : 1,000 以下
```

**Open X-Embodimentの限界**:
- データ品質が不均一（機関によってバラバラ）
- 22ロボットとはいえ、多くはテーブルトップ操作に偏重
- 行動空間の定義が統一されていない

**データ品質 vs 量のトレードオフ**:
- π0は少数の高品質自社データで学習して高性能を達成
- OpenVLAはOpen X-Embodimentの大量データで学習したが質が問題に
- **品質のほうが重要**という傾向が見えている

### 7.3 行動表現の課題

- **離散トークン化の精度限界**: 256ビンでは精密操作に不十分。FASTで改善されつつある
- **マルチモーダル行動分布**: 同じ状況で複数の正解行動がある場合（例: 左から回り込むか右から回り込むか）、離散化やMSE損失ではmode collapseが起きる
- **行動チャンクの長さ**: 短すぎると反応的、長すぎると環境変化に対応できない

### 7.4 評価の課題

- **既存ベンチマークの限界**: LIBERO-PRO/Plusが指摘した通り、標準ベンチマークのスコアは実力を反映していない
- **実機評価の再現性**: ロボットの機体差、環境の微妙な違いで再現困難
- **何を測るべきかの未合意**: success rate? robustness? データ効率? 推論速度? 汎化の軸をどう分離する?
- **VLANeXt**（[arXiv:2602.18532](https://arxiv.org/abs/2602.18532)）は LIBERO + LIBERO-Plus の両方で評価しており、**摂動込み評価が新しい標準になりつつある**

---

## 8. 2025年の注目トレンド

### 8.1 World Models + VLA

- **GR-2**の発想: ビデオ生成（次フレーム予測）を中間表現として行動を生成
- 「物理世界のシミュレーション能力」を獲得することで汎化性能を向上させる狙い
- Sora/Video Generation modelの技術がロボティクスに流入中

### 8.2 RL + VLA（オンライン学習の導入）

- **SimpleVLA-RL**: VLAをRLで追加学習し、実環境でのフィードバックを取り込む
- 従来のVLAはオフラインデータ（デモ）のみで学習 → distributional shiftの根本解決にはオンライン学習が必要
- RLHF/RLAIFのロボティクス版として注目

### 8.3 空間理解の強化

- **SpatialVLA**: 3D空間情報をVLAに組み込む。深度推定 + 3D位置エンコーディング
- VLAは本質的に「画像を見ている」だけで「空間を理解している」わけではないことが LIBERO-PROで露呈
- 空間表現の強化は丸暗記問題への直接的な対策

### 8.4 効率化

- **VLA-Perf**: VLA推論性能のプロファイリングとボトルネック分析
- **推論速度**: ロボット制御には10Hz以上が望ましいが、7Bモデルの推論はGPU無しでは困難
- **fine-tune効率**: LoRA, QLoRA などのPEFT手法のVLAへの適用

### 8.5 設計レシピの体系化

- **VLANeXt**: 12の設計知見を実験的に検証。**「何が効いて何が効かないか」のエビデンスベースの指針**
- VLMバックボーンの選択（LLaMA + SigLIP2が現時点で最良）
- 行動表現の選択
- 学習戦略（事前学習→fine-tuneのパイプライン）

---

## 9. コンペ運営者として押さえるべきポイント

### 9.1 「なぜ既存の評価では不十分か」を語れること

> 「VLAは標準LIBEROで90%以上達成しているが、LIBERO-PROが示した通り、物体位置をわずかにずらすだけで0%に崩壊する。これは丸暗記であり、真の汎化ではない。」

この一文を根拠データ付きで説明できることが、評価プロトコルの正当性の根幹。

### 9.2 参加者に理解させるべき技術スタック

| レイヤー | 内容 | Phase |
|---------|------|-------|
| VLMバックボーン | PaLI-Gemma, LLaMA + SigLIP | 全Phase |
| 行動表現 | 離散トークン / Flow Matching / FAST | Phase 1, 3 |
| 学習パイプライン | openpiでのfine-tune手順 | Phase 1 |
| 評価環境 | LIBERO + LIBERO-PRO | Phase 1, 2 |
| 問題分析 | なぜ崩壊するかの構造的理解 | Phase 2 |
| 改善手法 | データ拡張, ドメインランダマイゼーション等 | Phase 3 |

### 9.3 自分が先に触っておくべき順序

1. **openpi + LIBERO のセットアップ** → Docker で評価を通しで回す
2. **π0-FAST の fine-tune** → LIBERO データで学習し、4スイートで評価
3. **LIBERO-PRO の摂動実験** → 自分のfine-tuneモデルで Position/Instruction 摂動を確認
4. **OpenVLA との比較** → 参加者が使う可能性のある別モデルでの結果を把握
5. **VLANeXt の論文精読** → 最新の設計知見を理解し、参加者の質問に答えられるように

### 9.4 想定QA（参加者から聞かれそうな質問）

| 質問 | 回答のポイント |
|------|-------------|
| 「なぜ離散化ではなくFlow Matchingを使うのか？」 | 精度の問題。256ビンでは精密操作に限界。FAST等の中間的アプローチも存在する |
| 「データ拡張はどの軸が効くのか？」 | Position摂動が最もクリティカル（LIBERO-PROの結果）。色・テクスチャは効果限定的 |
| 「なぜ言語を無視するのか？」 | 学習データで同一シーン・同一タスクのペアが多く、言語なしでも視覚だけで正解軌道を特定できてしまう |
| 「π0.5がπ0より強いのはなぜ？」 | Webデータとの統合により、より広い視覚・言語理解を獲得。ただし位置摂動には依然脆弱 |

---

## 参考文献（主要論文リスト）

### VLA以前
- Ross et al., "DAgger" (2011) — 分布シフトの理論的解法
- Ho & Ermon, "GAIL" (2016) — 敵対的模倣学習
- Radford et al., "CLIP" (2021) — 視覚-言語共通表現
- Dosovitskiy et al., "ViT" (2021) — Vision Transformer
- Ahn et al., "SayCan" (2022) — LLMの物理世界接地
- Liu et al., "LLaVA" (2023) — オープンソースVLM

### VLAモデル
- Brohan et al., [RT-2](https://arxiv.org/abs/2307.15818) (2023)
- Ghosh et al., [Octo](https://arxiv.org/abs/2405.12213) (2024)
- Kim et al., [OpenVLA](https://arxiv.org/abs/2406.09246) (2024)
- Black et al., [π0](https://arxiv.org/abs/2410.24164) (2024)
- [SpatialVLA](https://arxiv.org/abs/2501.15830) (2025)
- [VLANeXt](https://arxiv.org/abs/2602.18532) (2025)

### 評価・課題
- Bo Liu et al., "LIBERO" (NeurIPS 2023) — lifelong learning ベンチマーク
- Zhou et al., [LIBERO-PRO](https://arxiv.org/abs/2510.03827) (2025) — 丸暗記問題の暴露
- [LIBERO-Plus](https://arxiv.org/abs/2510.13626) (2025) — 7次元摂動分析
- Brohan et al., "RT-1" (2023) — 大規模ロボットTransformerの実証
