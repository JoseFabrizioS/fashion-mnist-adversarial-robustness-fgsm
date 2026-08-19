# Dataset Fashion-MNIST

Esta carpeta documenta el dataset utilizado por el proyecto. Los archivos con las 70,000 imágenes no se almacenan en GitHub porque TensorFlow/Keras los descarga automáticamente desde su fuente pública y los conserva en una caché local.

## Fuente y mecanismo de descarga

- **Fuente original:** [Fashion-MNIST de Zalando Research](https://github.com/zalandoresearch/fashion-mnist).
- **Cargador utilizado:** [`tf.keras.datasets.fashion_mnist.load_data()`](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data).
- **Alojamiento utilizado por Keras:** `https://storage.googleapis.com/tensorflow/tf-keras-datasets/`.

El código empleado en el proyecto es:

```python
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)
```

La primera ejecución descarga cuatro archivos comprimidos: imágenes y etiquetas de entrenamiento, además de imágenes y etiquetas de prueba. Las siguientes ejecuciones reutilizan la copia guardada en la caché.

## División del dataset

| Variable | Contenido | Dimensión original |
|---|---|---:|
| `x_train` | Imágenes de entrenamiento | `(60000, 28, 28)` |
| `y_train` | Etiquetas de entrenamiento | `(60000,)` |
| `x_test` | Imágenes de prueba | `(10000, 28, 28)` |
| `y_test` | Etiquetas de prueba | `(10000,)` |

Cada imagen es de `28 x 28` píxeles, está en escala de grises y originalmente contiene valores enteros entre 0 y 255.

## Categorías

| Etiqueta | Categoría |
|---:|---|
| 0 | Camiseta/top |
| 1 | Pantalón |
| 2 | Suéter |
| 3 | Vestido |
| 4 | Abrigo |
| 5 | Sandalia |
| 6 | Camisa |
| 7 | Zapatilla |
| 8 | Bolso |
| 9 | Botín |

## Preparación utilizada por la CNN

```python
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
```

La normalización transforma los píxeles al rango `[0, 1]`. El nuevo eje representa el único canal de la imagen, por lo que las entradas finales tienen dimensiones `(60000, 28, 28, 1)` y `(10000, 28, 28, 1)`.

## Visualización de una muestra

Este bloque puede ejecutarse después de cargar y preparar el dataset:

```python
import matplotlib.pyplot as plt

class_names = [
    "Camiseta", "Pantalón", "Suéter", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín",
]

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i].squeeze(), cmap="gray")
    plt.title(class_names[int(y_train[i])])
    plt.axis("off")

plt.tight_layout()
plt.show()
```

## Ubicación en Google Colab

`sample_data` es una carpeta de ejemplos incluida por Colab y no contiene Fashion-MNIST. El dataset descargado por Keras normalmente se encuentra en:

```text
/root/.keras/datasets/fashion-mnist/
```

Puede comprobarse desde una celda de Colab con:

```python
!ls -lh /root/.keras/datasets/fashion-mnist/
```

En una computadora local, Keras utiliza normalmente `~/.keras/datasets/fashion-mnist/` dentro de la carpeta del usuario.

## Política del repositorio

El archivo `.gitignore` excluye cualquier dato descargado dentro de `data/` y conserva únicamente este documento. Esto mantiene el repositorio ligero y evita duplicar un dataset público que ya cuenta con un mecanismo oficial de descarga.
