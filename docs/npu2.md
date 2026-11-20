# NPU2 变体（`npu2` 分支）

`main` 分支的模型按 **NPU3/NPU1 混合**编译（VAE 用 3 核，LM/DiT 用 1 核）。
本分支把 **9 个子图统一改为 `npu_mode = NPU2`** 重新编译，便于和 ASR/其他模型共卡
（AX650 共 3 个 NPU 核，NPU2 只占 2 个）。

## 差异

| 项 | main | npu2 |
|---|---|---|
| `npu_mode` | vae_enc / vae_dec_* / vae_dec_stream = NPU3；lm_* / dit / locenc / spk = NPU1 | 全部 **NPU2** |
| 量化 | U16 激活 / S8 权重 / U16 输出 + SmoothQuant | 相同 |
| 校准集 | 相同 | 相同 |
| 模型精度 | 基准 | 见下表（板端实测） |
| 速度 | 基准 | 见下表（板端实测） |

## 重新编译

```bash
# 1) 生成 NPU2 配置（config 里 npu_mode 全部为 NPU2）
python model_convert/prepare_compile.py --npu NPU2 --compile-dir compile_npu2

# 2) 逐个编译（9 个子图，可并行）
for n in spk_body locenc_step dit_step lm_prefill lm_decode vae_dec_l80 vae_dec_l128 vae_dec_l256 vae_enc; do
  pulsar2 build --config compile_npu2/pulsar2_config_${n}.json > compile_npu2/logs/${n}.log 2>&1 &
done
wait
```

## 板端结果

| 子图 | 精度（相对 FP32，板端） |
|---|---|
| vae_enc | cosine 0.999873 / 35.90 dB |
| spk_body | cosine 0.999615 / 31.02 dB |
| locenc_step | cosine 0.999988 / 45.93 dB |
| lm_prefill（hidden last） | cosine 0.999621 / 31.20 dB |
| lm_decode | cosine 0.999662 / 31.62 dB |
| dit_step（4 步累计） | cosine 0.997661 / 23.19 dB |
| vae_dec_l80 / l128 / l256 | 29–30 dB（l128 29.18 / l256 29.51） |

> 与 `main`（NPU3/NPU1 混合）**逐图数值完全一致**：同样的量化与图，只是核心数不同。

| 场景（C++） | TTFT（首 latent） | TTFA（首音频） | 总耗时 | RTF |
|---|---|---|---|---|
| 流式 3.04 s | 0.12–0.13 s | 0.48–0.49 s | 1.95 s | 0.64 |
| 流式 10.24 s | 0.13 s | 0.49 s | 4.94–4.95 s | 0.48 |
| 非流式 3.04 s | — | 1.00–1.01 s | 1.00–1.01 s | 0.33 |

对比 `main`（NPU3）：流式 3.04 s 为 `TTFT 0.15 / TTFA 0.47 / RTF 0.61`，
10.24 s 为 `RTF 0.46`，非流式 `RTF 0.35`。
NPU2 在流式解码（l80 窗口，3→2 核）上慢约 3~6%，但 LM/DiT（1→2 核）首包更快，
非流式反而更快（`0.349 → 0.331`）。

音频与 `main` 分支的对比：**md5 完全相同**（同一 seed、同一文本），即逐样本一致，无精度损失。
