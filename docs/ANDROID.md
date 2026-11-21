# Usar el decoder de Alya en Android

Guía práctica para integrar `alya_decoder_int8.onnx` (11.89 M params, ~12 MB) en el APK.

## 1. Dependencia

```gradle
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
}
```

`onnxruntime-android` incluye XNNPACK + NNAPI. Si quieres el binario más pequeño,
usa `onnxruntime-mobile` y compila un `.ort` con las mismas ops.

## 2. Copiar el modelo

```
app/src/main/assets/alya_decoder_int8.onnx
```

En `build.gradle` evita que se comprima (empeora el arranque):

```gradle
androidResources { noCompress += "onnx" }
```

## 3. Sesión

```kotlin
import ai.onnxruntime.*
import java.nio.FloatBuffer

class AlyaDecoder(ctx: android.content.Context) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val bytes = ctx.assets.open("alya_decoder_int8.onnx").use { it.readBytes() }
        val opts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(2)          // T606: 2 hilos es el punto dulce
            addNnapi()                       // cae a CPU solo si NNAPI no soporta una op
        }
        session = env.createSession(bytes, opts)
    }

    /** z = FloatArray(192) {0f}  ->  cara canónica determinista (sin flicker) */
    fun render(z: FloatArray, labels: FloatArray): FloatArray {
        env.createTensorFromBuffer(FloatBuffer.wrap(z), longArrayOf(1, 192))
            .use { zt ->
                env.createTensorFromBuffer(FloatBuffer.wrap(labels), longArrayOf(1, 29))
                    .use { lt ->
                        val res = session.run(mapOf("z" to zt, "labels" to lt))
                        val out = res[0].value as java.util.HashMap<*, *>   // OnnxValue -> array
                        res.close()
                        return out["rgba"] as FloatArray                  // [1,4,128,128]
                    }
            }
    }

    override fun close() { session.close() }
}
```

> La salida viene en `[-1, 1]`. Para pintar: `v = (out * 0.5f + 0.5f)`.

## 4. De float[] a Bitmap (RGBA premultiplicado)

```kotlin
fun rgbaToBitmap(out: FloatArray, w: Int = 128, h: Int = 128): android.graphics.Bitmap {
    val px = IntArray(w * h)
    val plane = w * h
    for (i in 0 until plane) {
        val r = ((out[i] * 0.5f + 0.5f).coerceIn(0f, 1f) * 255).toInt()
        val g = ((out[plane + i] * 0.5f + 0.5f).coerceIn(0f, 1f) * 255).toInt()
        val b = ((out[2 * plane + i] * 0.5f + 0.5f).coerceIn(0f, 1f) * 255).toInt()
        val a = ((out[3 * plane + i] * 0.5f + 0.5f).coerceIn(0f, 1f) * 255).toInt()
        px[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
    }
    return android.graphics.Bitmap.createBitmap(px, w, h, android.graphics.Bitmap.Config.ARGB_8888)
}
```

## 5. Bucle de animación sin perder frames

El coste de generar 128² es bajo; el cuello real es la subida a 720p y el compositing.

* Genera el frame **en 128×128** y escala con un `SurfaceView` + shader (o `Canvas` con
  `FILTER_BITMAP`). No generes a 720p en CPU.
* Cachea los frames que dependen solo de `pose` + `emotion` (10–15 frames); durante el habla
  solo cambia `viseme`, así que puedes reutilizar el resto.
* Corre la inferencia en un hilo propio con `Dispatchers.Default` y entrega los bitmaps a la UI
  con doble buffer; nunca bloquees el hilo principal.
* Vigila la temperatura: a partir de ~45 °C baja a 30 fps y reduce el `intraOpNumThreads` a 1.

## 6. Etiquetas

El vector tiene 29 floats: `emotion[0:10] + pose[10:14] + viseme[14:24] + eyes[24:28] + intensity[28]`.
One-hot con 1.0 en el índice elegido. Ver `labels.json` en la raíz del repo.

## 7. Validación rápida antes de empaquetar

```bash
python -m onnxruntime.tools.check_onnx_model alya_decoder_int8.onnx   # estructura
```

Y en el teléfono, mide una vez al arrancar: si el primer frame tarda >100 ms, activa
`addNnapi()` explícitamente y compara; en T606 NNAPI suele ganar en las convoluciones de 128².
