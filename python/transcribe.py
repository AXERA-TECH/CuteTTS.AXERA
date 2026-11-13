#!/usr/bin/env python3
"""Reference-only helper: SenseVoice ASR on AX650 (libax_asr_api) for QA."""
import ctypes, os, sys
import numpy as np

_BASE = os.environ.get("ASR_BASE", "/root/huyuan/183_workspace/Speech-Translation-ZipVoice")
LIB = os.environ.get("ASR_LIB", _BASE + "/third_party/asr/lib/libax_asr_api.so")
MODEL = os.environ.get("ASR_MODEL", _BASE + "/models/asr_official/sensevoice_ax650")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cutetts_ax import audio as A

def main():
    lib = ctypes.CDLL(LIB)
    lib.AX_ASR_Init.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.AX_ASR_Init.restype = ctypes.c_void_p
    lib.AX_ASR_RunPCM.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                                 ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p)]
    lib.AX_ASR_RunPCM.restype = ctypes.c_int
    lib.AX_ASR_Free.argtypes = [ctypes.c_char_p]
    os.environ.setdefault("AX_SENSEVOICE_SHARE_DIR", MODEL)
    h = lib.AX_ASR_Init(4, MODEL.encode())
    assert h, "init failed"
    for path in sys.argv[1:]:
        wav, sr = A.read_wav(path)
        if sr != 16000:
            wav = A.resample_to(wav, sr, 16000)
        arr = np.ascontiguousarray(wav, dtype=np.float32)
        out = ctypes.c_char_p()
        rc = lib.AX_ASR_RunPCM(h, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), arr.size,
                              16000, b"auto", ctypes.byref(out))
        txt = out.value.decode("utf-8", "replace") if out.value else ""
        if out.value:
            lib.AX_ASR_Free(out)
        print(f"{os.path.basename(path)}: rc={rc} {txt.strip()}")

if __name__ == "__main__":
    main()
