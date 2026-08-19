# Guion de exposición — diapositivas 7 a 13

Este guion acompaña la presentación final y está diseñado para explicar con claridad la metodología, los resultados y las conclusiones del experimento.

## Diapositiva 7 — Funcionamiento de FGSM

FGSM significa *Fast Gradient Sign Method*. El procedimiento parte de una imagen limpia y calcula el gradiente de la pérdida con respecto a los píxeles de entrada, no con respecto a los pesos del modelo. El signo del gradiente indica la dirección en la que debe modificarse cada píxel para aumentar la pérdida. Esa dirección se multiplica por epsilon, que limita la intensidad de la perturbación. Finalmente, `clip` conserva los valores dentro del rango normalizado `[0, 1]`.

La CNN permanece fija durante el ataque. FGSM genera una copia perturbada de la imagen; no modifica los pesos ni el dataset original. El objetivo es comprobar si una imagen visualmente similar puede desplazar la decisión del modelo.

**Transición:** Una vez definido cómo se construye la imagen adversaria, revisaremos el protocolo aplicado al conjunto de prueba.

## Diapositiva 8 — Diseño experimental

Se entrena una sola CNN con semilla 42 y callbacks de estabilidad. Después se mide su accuracy limpia y se conserva el mismo modelo para siete evaluaciones adversarias. Las únicas entradas que cambian son las copias perturbadas con epsilon `0.00`, `0.01`, `0.03`, `0.05`, `0.10`, `0.20` y `0.30`.

La accuracy robusta representa el porcentaje de imágenes perturbadas que continúan clasificándose correctamente. La ASR considera únicamente las imágenes inicialmente correctas y mide qué proporción pasa a clasificarse incorrectamente después del ataque. Epsilon cero funciona como control: debe reproducir la accuracy limpia y producir una ASR igual a cero.

**Transición:** Antes de analizar la vulnerabilidad, debemos establecer el desempeño de la CNN en condiciones normales.

## Diapositiva 9 — Resultado limpio

La CNN obtuvo una accuracy limpia de `92.56%` en el conjunto de prueba. Este valor proviene de la ejecución documentada y se ubica en el extremo superior del rango recopilado para modelos comparables de Fashion-MNIST.

La comparación con benchmarks es contextual: los modelos públicos pueden usar configuraciones o preprocesamientos diferentes. El punto central es demostrar que la CNN clasifica bien las imágenes limpias. FGSM no busca elevar ese resultado, sino comprobar si las predicciones se mantienen estables ante perturbaciones dirigidas.

**Transición:** Con la línea base establecida, podemos observar cómo cambia la robustez al aumentar epsilon.

## Diapositiva 10 — Tendencia adversaria

La línea de accuracy robusta desciende en cada nivel, mientras la ASR aumenta. Con epsilon `0.05`, la robustez cae a `42.78%` y la ASR llega a `53.78%`: más de la mitad de los aciertos iniciales son revertidos. Con epsilon `0.30`, la accuracy robusta es `6.57%` y la ASR alcanza `93.06%`.

Esto demuestra que una accuracy limpia alta no garantiza estabilidad adversaria. Epsilon `0.30` debe interpretarse como una prueba de estrés fuerte y puede generar cambios perceptibles; por eso se informa toda la curva y no únicamente el valor extremo.

**Transición:** La gráfica muestra la tendencia general; la tabla siguiente permite revisar los valores exactos.

## Diapositiva 11 — Lectura de resultados

Con epsilon cero, la robustez coincide con `92.56%` y la ASR es `0%`, lo que funciona como comprobación interna de coherencia. Con `0.01` aparece una vulnerabilidad inicial; con `0.03` la robustez baja a `59.57%`; con `0.05` se invierte más de la mitad de los aciertos originales. Desde `0.10` el deterioro es severo y con `0.30` el modelo conserva solo `6.57%` de accuracy robusta.

La accuracy robusta y la ASR no son la misma métrica. La primera evalúa el desempeño sobre las entradas perturbadas; la segunda se condiciona a los ejemplos inicialmente correctos. El control con epsilon cero respalda la implementación, pero no constituye por sí solo una validación completa de seguridad.

**Transición:** Estos resultados permiten formular las conclusiones principales del trabajo.

## Diapositiva 12 — Conclusiones

La CNN presenta buen desempeño limpio y obtiene `92.56%`. La evaluación interna es consistente porque epsilon cero reproduce ese resultado. Sin embargo, la vulnerabilidad crece de forma monotónica con epsilon y la ASR llega a `93.06%`.

FGSM cumple una función diagnóstica: revela una fragilidad que la evaluación convencional no muestra. No es una defensa ni una técnica destinada a aumentar la accuracy limpia. El alcance del estudio está limitado a una arquitectura, un dataset y un ataque de un solo paso; por ello, los resultados describen vulnerabilidad frente a FGSM bajo este protocolo y no frente a todos los ataques posibles.

**Transición:** El diagnóstico ya está realizado; la siguiente etapa debe incorporar y validar una defensa.

## Diapositiva 13 — Siguiente fase

La mejora propuesta no consiste simplemente en agregar capas. Primero se plantea entrenamiento adversarial, mezclando ejemplos limpios y FGSM durante el entrenamiento. Segundo, se propone validar con PGD, un ataque iterativo más exigente. Tercero, se debe comparar la CNN estándar y la CNN defendida bajo la misma arquitectura, semillas y protocolo, utilizando accuracy limpia, accuracy robusta y ASR.

El mensaje final es que evaluar solamente la accuracy limpia no es suficiente: un modelo puede clasificar bien y, al mismo tiempo, ser sensible a perturbaciones construidas a partir de sus propios gradientes.

## Preguntas frecuentes

### ¿FGSM modifica el modelo?

No. Durante la evaluación los pesos permanecen fijos; se modifica una copia de la imagen.

### ¿Por qué FGSM no aumenta la accuracy?

Porque es un mecanismo de diagnóstico adversario, no una técnica de optimización ni una defensa.

### ¿Qué representa epsilon?

La intensidad máxima permitida para la perturbación de cada píxel normalizado.

### ¿La ASR es igual a 100 menos la accuracy robusta?

No necesariamente. La ASR se calcula únicamente sobre los ejemplos inicialmente correctos.

### ¿El modelo ya está defendido?

No. El trabajo actual diagnostica la vulnerabilidad. La defensa corresponde a una etapa posterior de entrenamiento adversarial y validación con ataques más fuertes.
