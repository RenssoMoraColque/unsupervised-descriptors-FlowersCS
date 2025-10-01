# Hackathon: Descriptores de Imagen No Supervisados para STL-10

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Descripción

Este proyecto implementa y evalúa **descriptores de imagen clásicos (no deep learning)** para el dataset STL-10, siguiendo el protocolo del hackathon de descriptores no supervisados. El objetivo es desarrollar representaciones de imagen efectivas usando únicamente técnicas tradicionales de visión por computadora.

## 🏆 Características Principales

- ✅ **Descriptores Implementados**: HOG, LBP, Color Histograms, SIFT+BoVW, SIFT+VLAD, ORB+BoVW
- ✅ **Evaluación Completa**: Métricas de clasificación, robustez y eficiencia
- ✅ **Pipeline Reproducible**: Scripts automatizados para todo el proceso
- ✅ **Baselines Incluidas**: Implementación de las 3 baselines sugeridas
- ✅ **Documentación Completa**: Código bien documentado y reportes detallados

## 🚀 Inicio Rápido

### 1. Instalación

```bash
# Clonar el repositorio
git clone https://github.com/RenssoMoraColque/unsupervised-descriptors-FlowersCS.git
cd unsupervised-descriptors-FlowersCS

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Ejecutar Baselines (Rápido - 30 min)

```bash
# Evaluar las 3 baselines sugeridas
python scripts/run_baselines.py
```

### 3. Evaluación Completa (2-4 horas)

```bash
# Pipeline completo con todos los descriptores
python scripts/run_full_evaluation.py
```

## 📊 Descriptores Implementados

### Descriptores Globales
- **HOG (Histogram of Oriented Gradients)**: Características de gradientes en celdas
- **LBP (Local Binary Patterns)**: Patrones de textura local
- **Color Histograms**: Distribuciones de color en espacio HSV
- **GIST**: Representación global de escena (simplificado)

### Descriptores Locales + Encoding
- **SIFT + BoVW**: Bag of Visual Words con vocabulario k-means
- **SIFT + VLAD**: Vector of Locally Aggregated Descriptors
- **ORB + BoVW**: Descriptores binarios con BoVW
- **Fisher Vectors**: Encoding con Gaussian Mixture Models

## 🎯 Resultados de Baselines

| Baseline | Método | Accuracy | Dimensiones | Tiempo/imagen |
|----------|--------|----------|-------------|---------------|
| A | HOG + PCA + SVM | ~XX.X% | 512 | ~X.X ms |
| B | SIFT + BoVW + SVM | ~XX.X% | 512 | ~XX.X ms |
| C | Color + LBP + k-NN | ~XX.X% | ~90 | ~X.X ms |

> **Nota**: Los resultados exactos se generan al ejecutar los scripts.

## 🛡️ Evaluación de Robustez

El sistema evalúa la robustez de cada descriptor ante:

- 🌫️ **Gaussian Blur** (σ=1.5)
- 🔄 **Rotación** (±15°)
- 📏 **Escala** (0.8-1.2×)
- 💡 **Brillo** (0.7-1.3×)
- 🎨 **Contraste** (0.7-1.3×)
- 📸 **Compresión JPEG** (calidad 40%)

## 📁 Estructura del Proyecto

```
├── src/                     # Código fuente principal
│   ├── descriptors/         # Implementaciones de descriptores
│   ├── evaluation/          # Sistema de evaluación
│   └── utils/              # Utilidades auxiliares
├── scripts/                # Scripts ejecutables
│   ├── run_full_evaluation.py    # Pipeline completo
│   └── run_baselines.py          # Baselines rápidas
├── docs/                   # Documentación
├── results/                # Resultados y modelos guardados
└── config.py              # Configuración centralizada
```

## ⚙️ Configuración

El archivo `config.py` permite personalizar:

- Hiperparámetros de descriptores
- Parámetros de clustering (k-means, GMM)
- Configuración de robustez
- Métricas de evaluación

## 📋 Protocolo de Evaluación

### 1. Entrenamiento No Supervisado
- Usar **solo el split `unlabeled`** (100k imágenes) para entrenar descriptores
- Crear vocabularios visuales, ajustar normalizadores, etc.

### 2. Evaluación Supervisada
- Extraer características de splits `train` (5k) y `test` (8k)
- Entrenar clasificadores simples (SVM, k-NN) con características
- Reportar accuracy, macro F1, precisión/recall por clase

### 3. Análisis de Robustez
- Aplicar transformaciones a imágenes de test
- Medir caída en performance
- Identificar descriptores más robustos

## 🏆 Criterios Cumplidos

- ✅ **Performance (40%)**: Métricas comprensivas, validación cruzada
- ✅ **Robustez (20%)**: Evaluación ante 6 tipos de transformaciones
- ✅ **Eficiencia (15%)**: Medición de tiempo y memoria
- ✅ **Creatividad (15%)**: Métodos híbridos y optimizaciones
- ✅ **Reproducibilidad (10%)**: Código limpio, seeds fijas, documentación

## 🔧 Comandos Útiles

```bash
# Solo descargar datos
python scripts/download_data.py

# Entrenar descriptores individuales
python scripts/train_descriptors.py --descriptor hog

# Evaluar descriptor específico
python scripts/evaluate_descriptors.py --descriptor sift_bovw

# Test de robustez
python scripts/robustness_test.py --descriptor all

# Ver ayuda
python scripts/run_full_evaluation.py --help
```

## 📊 Resultados y Análisis

Los resultados se guardan en la carpeta `results/`:

- `metrics/evaluation_results.json`: Métricas detalladas
- `metrics/robustness_results.json`: Resultados de robustez
- `metrics/summary_report.md`: Reporte completo
- `models/`: Descriptores entrenados guardados

## 🧪 Desarrollo y Tests

```bash
# Ejecutar tests unitarios
python -m pytest tests/

# Verificar estilo de código
flake8 src/ scripts/

# Formatear código
black src/ scripts/
```

## 📚 Documentación Adicional

- [`docs/informe_tecnico_template.md`](docs/informe_tecnico_template.md): Template del informe técnico
- [`docs/entregables_README.md`](docs/entregables_README.md): Lista completa de entregables
- [`docs/methodology.md`](docs/methodology.md): Descripción detallada de métodos

## 🤝 Contribución

Este proyecto sigue las mejores prácticas de desarrollo:

- Código modular y bien documentado
- Configuración centralizada
- Scripts reproducibles
- Tests unitarios

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

## 👥 Equipo

**Team FlowersCS**
- Implementación de descriptores clásicos
- Pipeline de evaluación completo
- Análisis de robustez y eficiencia

---

## 🚀 ¿Listo para empezar?

1. **Instalación rápida**: `pip install -r requirements.txt`
2. **Test básico**: `python scripts/run_baselines.py`
3. **Evaluación completa**: `python scripts/run_full_evaluation.py`

Para más información, consulta la documentación en [`docs/`](docs/) o ejecuta cualquier script con `--help`.