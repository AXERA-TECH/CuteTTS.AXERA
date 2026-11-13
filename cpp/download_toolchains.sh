#!/usr/bin/env bash
# Fetch the AX650 BSP (ax_engine headers/libs) and the Arm GNU aarch64 toolchain.
set -euo pipefail
CPP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${CPP_DIR}/toolchains"
mkdir -p "$DEST"
: "${GH_PROXY:=}"
: "${BSP_URL:=https://github.com/AXERA-TECH/ax650n_bsp_sdk/archive/refs/heads/main.zip}"
: "${TOOLCHAIN_URL:=https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz}"
cd "$DEST"
if [[ ! -d ax650n_bsp_sdk ]]; then
  echo "== downloading BSP"
  curl -L --retry 3 -o bsp.zip "${GH_PROXY}${BSP_URL}"
  unzip -q bsp.zip && mv ax650n_bsp_sdk-main ax650n_bsp_sdk && rm -f bsp.zip
fi
if [[ ! -x gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++ ]]; then
  echo "== downloading toolchain"
  curl -L --retry 3 -o tc.tar.xz "$TOOLCHAIN_URL"
  tar xf tc.tar.xz && rm -f tc.tar.xz
fi
echo "toolchains -> $DEST"
