# CuteTTS-distill 在 AX650 上的部署实现

## 1. 模型结构

CuteTTS 是"连续隐变量自回归 + 扩散头 + 声码器"的 TTS：

```
文本 ──tokenizer──> token ids ──embedding──┐
参考音频 ──VAE encoder──> 连续 latent ──LocEnc──┤
参考音频 ──log-mel──> Speaker Encoder ──线性投影─┴─> LM prefix
LM(Qwen3, 7 层) 自回归 1 token ≈ 2 latent ≈ 160 ms 音频
  └─ 每个 token 的 hidden ──AudioDiT(4 步 Euler)──> 2 帧 latent
latent ──VAE decoder──> 24 kHz 波形
```

CuteTTS-distill 与 base 的差别：扩散头是 4 步蒸馏版（单分支 distilled CFG，
`cfg_strength` 作为条件输入），不需要 LM 级 CFG，因此每步只跑一个分支。

| 子模型 | 参数量 | 作用 | 运行次数 |
|---|---|---|---|
| VAE encoder | 128 M | 参考音频 → 64 维 latent | 每次合成 1 次 |
| Speaker encoder | 26 M | log-mel → 256 维音色向量 | 每次合成 1 次 |
| LocEnc (+proj) | 13 M | latent patch → LM 输入 embedding | 1 次（参考）+ 每步 1 次 |
| LM prefill | 150 M | 256 token 前缀 | 1 次 |
| LM decode | 150 M | 1 token + KV cache | 每步 1 次 |
| AudioDiT step | 74 M | 单步 Euler 速度场 | 每步 4 次 |
| VAE decoder | 128 M | latent → 波形 | 1 次（按 64/128/256 帧分桶） |

## 2. 关键设计

### 2.1 静态 shape 与分桶

Pulsar2 要求静态 shape，所有子图都按固定 shape 导出：

- VAE encoder：5 s @ 24 kHz = 120000 采样（因果卷积，末尾补零安全）
- Speaker encoder：8 s @ 16 kHz → 800 帧 log-mel
- LocEnc：prompt 128 patch（256 latent = 20.5 s）/ 单步 1 patch
- LM prefill：256 token（前缀上限，超出则拒绝）
- LM decode：KV 容量 384（前缀 256 + 最多 128 个新 token = 20.5 s 音频）
- VAE decoder：64 / 128 / 256 latent 帧三档（5.1 / 10.2 / 20.5 s）
- AudioDiT：单步 5 token（1 时间步 + 2 条件 + 2 latent），host 循环 4 次

VAE 编解码器都是因果结构，所以"末尾补零 + 截断"与逐块流式解码等价，
不需要实现跨块卷积状态。

### 2.2 LM 的 KV cache

Qwen3 解码器导出为两个图：

- `lm_prefill`：256 token 因果注意力，输出最后一层 hidden + 7 层的 K/V
- `lm_decode`：单 token，输入 384 长度的 K/V 缓存 + 加性 mask，输出新的 K/V

mask 用 0/-30 的加性形式（而不是 0/1 乘法后再归一化），避免量化后
softmax 数值下溢；量化范围 [-30, +30] 对 U16 也足够。

host 侧只把新 token 的 K/V 写回 numpy 缓存，每步上传 14×1.5 MB = 21 MB，
下载约 100 KB。AX650 上单步 decode 约 10–20 ms（含拷贝）。

### 2.3 音色条件

- 参考音频 12 s（可调）→ VAE latent → `(x + bias) * scale` 归一化后送 LocEnc
- speaker encoder 的 log-mel 在 host 用 numpy 计算（与 `torch.stft(center=True)`
  逐点一致，cosine = 1.0），NPU 只跑 ECAPA 主干
- `lm_speaker_linear`（256→1024）与 embedding 查表在 host 完成（embedding 表
  16.4 MB fp16），其余全部在 NPU

### 2.4 扩散头

4 步蒸馏 Euler 采样在 host 展开：`x ← x + v(x, z, cond, t, dt) * 0.25`，
每步一次 NPU 调用。相比把 4 步融进一个 ONNX 图，单步图只有 1/4 权重
（65 MB vs 263 MB），编译时间也从约 10 min 降到约 1 min。

### 2.5 量化

- 校准集：14 条真实语音（中英、6–48 kHz 重采样到 24 kHz），真实提示词
  embedding、真实 KV cache、真实扩散步输入（含首 patch，cond=0）
- `DEFAULT` 层：`data_type=U16` / `weight_data_type=S8` / `output_data_type=U16`，
  `calibration_method=MinMax`，`enable_smooth_quant=true`
- `precision_analysis=true` + `EndToEnd`，逐图对比板端输出
- `compiler.check=0`：不做仿真，精度全部在 AX650 板端验证

板端逐图精度（相对 FP32）：

| 子图 | 精度 |
|---|---|
| `vae_enc` | 35.9 dB |
| `spk_body` | 31.0 dB |
| `locenc_step` | 45.9 dB |
| `lm_prefill` / `lm_decode` | 31.2 / 31.6 dB |
| `dit_step`（4 步） | 23.2 dB（cos 0.9977） |
| `vae_dec_l80/l128/l256` | 29–30 dB |

## 3. C++ 运行时

`cpp/` 用 AX_ENGINE + STL 复刻同一套流程（不依赖 Python/numpy）：

- 9 个 axmodel 的加载 / 运行（`ax_runner.hpp`，按 tensor 名读写 float32）
- WAV 读写、polyphase 重采样、512 点 FFT log-mel（与 torch.stft 对齐）
- SentencePiece **BPE** 编码器（`assets/tokenizer.bin`，由 `export_assets.py` 导出）
- embedding 查表、speaker 线性投影、mask/RoPE/KV 拼接、4 步 Euler
- 板端实测：en_default 3.04 s 音频 / 1.48 s（RTF 0.49），比 Python SDK 快约 8%

## 4. 流式输出

最终交付只有一条解码路径：**固定 80-latent 滑动窗**（`vae_dec_l80`），流式与离线共用同一张图，
所以两者输出逐样本一致（cos 1.000000）。

- **非流式**：AR 跑完后按音频长度选桶（`vae_dec_l80/l128/l256`）一次性解码，RTF 最低
- **流式**：AR 不停，每攒够 16 latent（1.28 s）解码一次：窗口 `[t-80, t)`，
  前 64 个 latent 作因果上下文，只输出新增 16 个 latent 的采样点
- **首包**：只等 4 latent（0.32 s），窗口尾部补零 —— 因果性保证补在后面的零不影响
  已生成部分的输出，因此首包不必等满 16 个 latent

依据：DAC 解码器全部是因果卷积（无 attention、无可携带隐状态），实测感受野 = 60 latent
（把序列前 60 个 latent 清零，第 60 个之后的采样点逐样本不变）。因此窗口上下文 ≥64
即可复现整句离线解码。

已被放弃的两版方案（脚本保留在 `model_convert/`，供参考）：

- `vae_dec_stream`：把 31 个因果卷积的左侧上下文做成 18 个显式状态张量，
  每步只喂 2 个 latent（160 ms/包），算力最省；但板端状态被量化到 U16 后误差逐步累积，
  逐段 SNR 11.6 → 17.9 dB 漂移（离线段约 30 dB），听感有底噪
- `vae_dec_l64` 桶：与 l80 重叠且无优势，已删除

代价：每包重复计算 64 个 latent 的上下文，流式 RTF 0.61（3 s 音频）/ 0.49（10 s 音频），
非流式 0.35–0.41；换来的是与离线完全一致的音质和 0.48~0.54 s 首音频（TTFA）。

## 5. 复现

```bash
# 0) 环境
pip install torch transformers safetensors sentencepiece soundfile scipy onnx onnxruntime
#    Pulsar2 8.0：https://pulsar2-docs.readthedocs.io

# 1) 下载权重（上游 OPPOer/CuteTTS-distill）到 origin/
# 2) 导出 ONNX
python model_convert/export_onnx.py            # 全部子图
python model_convert/export_dit1.py            # 单步扩散头
python model_convert/export_assets.py          # embedding 表/mel 滤波器等
# 3) 校准集
python model_convert/make_calib.py
python model_convert/make_calib_dit1.py
# 4) 编译
python model_convert/prepare_compile.py --run
# 5) 板端
python python/demo.py --model-dir models --assets models/assets \
    --text "..." --reference samples/reference.wav --output out.wav
```
