# Alya Decoder — Stage 1 (on-device facial frame generator)

Modelo neuronal que **dibuja un retrato de Alya (cabeza y hombros) desde cero** a partir de
un vector de condiciones (emoción, pose, visema, ojos, intensidad). No usa warp ni flow:
genera la imagen completa.

* Arquitectura: decodificador latente condicional (FiLM + RMS-norm) con GAN auxiliar
* Salida: **RGBA 128×128**, alfa aprendido → sin fondo
* Parámetros: **11.89 M**
* Tamaño INT8: **12.14 MB** · FP32: **47.57 MB**
* Entrenado 14000 iteraciones sobre 1000 imágenes de Alya
* Inferencia 100 % local (ONNX Runtime), sin red

![demo](examples/demo_grid.png)

---

## 1. Instalación

```bash
pip install onnxruntime numpy pillow
```

## 2. Uso mínimo (Python)

```python
import numpy as np, onnxruntime as ort

sess = ort.InferenceSession("alya_decoder_int8.onnx", providers=["CPUExecutionProvider"])

def label(emotion="neutral", pose="frente", viseme="m", eyes="abiertos", intensity=0.7):
    EMO = ['neutral', 'sonrisa', 'alegre', 'triste', 'enojada', 'sorpresa', 'preocupada', 'pensativa', 'timida', 'guino']
    POS = ['frente', 'izquierda', 'derecha', 'inclina']
    VIS = ['m', 'a', 'e', 'i', 'o', 'u', 'fv', 'l', 's', 'smile_open']
    EYE = ['abiertos', 'cerrados', 'medio', 'lado']
    v = np.zeros(29, np.float32)
    if emotion in EMO: v[EMO.index(emotion)] = 1
    if pose in POS:    v[10 + POS.index(pose)] = 1
    if viseme in VIS:  v[14 + VIS.index(viseme)] = 1
    if eyes in EYE:    v[24 + EYE.index(eyes)] = 1
    v[28] = intensity
    return v

z = np.zeros((1, 192), np.float32)                 # z=0 -> salida determinista
out = sess.run(None, {"z": z, "labels": label("sonrisa", viseme="a")[None, :]})[0][0]

rgb = np.clip(out[:3] * 0.5 + 0.5, 0, 1)           # [3,128,128] 0..1
alpha = np.clip(out[3] * 0.5 + 0.5, 0, 1)          # [128,128]   0..1
```

`z` controla la variación de identidad/estilo. `z = 0` es la cara canónica (estable entre
frames, ideal para animación). Un `z` fijo pequeño (p. ej. `np.random.randn(1,192)*0.6`)
da variación sin romper la identidad.

## 3. Formato de las etiquetas (29 floats)

| índices | campo | tipo | valores |
|---|---|---|---|
| 0–9 | emoción | one-hot | neutral, sonrisa, alegre, triste, enojada, sorpresa, preocupada, pensativa, timida, guino |
| 10–13 | pose | one-hot | frente, izquierda, derecha, inclina |
| 14–23 | visema | one-hot | m, a, e, i, o, u, fv, l, s, smile_open |
| 24–27 | ojos | one-hot | abiertos, cerrados, medio, lado |
| 28 | intensidad | escalar | 0.0 – 1.0 |

Detalle completo en [`labels.json`](labels.json). Un campo sin usar se deja a cero.

## 4. Rendimiento medido

| modelo | tamaño | CPU (esta máquina) |
|---|---|---|
| FP32 | 47.57 MB | 61.0 ms/frame |
| FP16 | 23.82 MB | ~misma que FP32 |
| INT8 (QDQ) | 12.14 MB | 80.8 ms/frame |

INT8 cuesta algo de fidelidad: MAE `0.0257` frente a la salida FP32
(rango de salida ±1), con errores locales puntuales de hasta `1.42`.
FP16 es prácticamente idéntico al FP32 (MAE `0.00006`). Si ves artefactos en
INT8, usa `alya_decoder_fp16.onnx`.

Fidelidad numérica frente a PyTorch: FP32 error máximo `2.53e-06`,
INT8 MAE `0.0257` (salida en rango ±1).

En un Unisoc T606 las estimaciones conservadoras son 3–6× más lentas que este x86;
con XNNPACK/NNAPI en Android el objetivo es mantener 30 fps generando cada frame y
usando caché de 10–15 frames para los que solo cambia la boca.

## 5. Android (ONNX Runtime Mobile)

```kotlin
val opts = SessionOptions().apply { addNnapi(); setIntraOpNumThreads(2) }
val env  = OrtEnvironment.getEnvironment()
val sess = env.createSession(assetPath, opts)   // alya_decoder_int8.onnx en assets/
```

Guía ampliada en [`docs/ANDROID.md`](docs/ANDROID.md).

## 6. Entrenamiento

* Dataset: 1000 renders de Alya (768²) → recorte a cabeza y hombros 256², RGBA
* Pérdidas: L1 de píxel + L1 de bordes (lineart) + KL (VAE) + GAN hinge auxiliar (peso 0.1, R1)
* 128², batch 32, Adam, ~14000 iteraciones, T4
* Última línea del log: `it 13900/14000 | rec 0.1368 | edg 0.3197 | kl 10.219 | D 0.630 | 120 img/s | eta 0 min`

## 7. Limitaciones honestas (Stage 1)

* Resolución nativa 128×128; el upscaler a 720p es el módulo 3 y aún no está.
* Una sola identidad (Alya). No generaliza a otras caras.
* Movimiento entre frames es limitado: la parte temporal (módulo 2) no está entrenada todavía.
  Las poses grandes pueden producir artefactos.
* Es un modelo base entrenado con 1000 imágenes; con más datos y más iteraciones
  mejora de forma directa.

## 8. Hoja de ruta

1. **Stage 2**: predictor temporal (audio MFCC → latentes por frame o visemas directos) para lipsync real.
2. **Stage 3**: upscaler 128² → 720p (ultraligero, INT8) + calibración de color.
3. Emparejado con TTS Piper-light en el APK universal.

## Licencia

Pesos y código: uso personal/experimental. La identidad de Alya pertenece a su autor.
