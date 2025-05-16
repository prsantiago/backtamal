# 📦 Backtamal – Un autodiff minimalista con sabor mexicano 🌽💻

**Backtamal** es una biblioteca educativa y minimalista de diferenciación automática (autodiff), inspirada en micrograd de Andrej Karpathy, pero con un twist especial: compilación opcional con Cython para acelerar operaciones críticas.

---

## 🔧 Características principales

- Implementación clara desde cero de autodiff basado en grafos.
- Operadores matemáticos básicos (`+`, `-`, `*`, `/`, `**`) con retropropagación.
- Soporte para funciones no lineales como `tanh`, `exp`, `relu`, etc.
- Compilación con Cython para mejorar el rendimiento de `Valor` y otras funciones clave.
- Código comentado y pensado para aprendizaje y experimentación.

---

## 📚 Ideal para

- Estudiantes que quieren entender cómo funciona el backpropagation internamente.
- Desarrolladores curiosos que quieren aprender autodiff sin frameworks complejos.
- Hackers que quieren experimentar con compilación acelerada en Python.

---

## 🧠 ¿Por qué "Backtamal"?

Porque es un sistema de backpropagation hecho en casa, con cariño y sabor local 🌽. Tan compacto como un tamal bien amarrado, pero poderoso por dentro.

---

## 🚀 ¿Cómo usar Backtamal?

### 1. Instalación

Clona el repositorio:

```bash
git clone https://github.com/prsantiago/backtamal.git
cd backtamal
```

### 2. Uso básico (versión Python puro)

Puedes importar y usar Backtamal directamente en Python:

```python
from backtamal.engine import Valor

a = Valor(2.0)
b = Valor(3.0)
c = a * b + a
c.retropropagacion()
print(a.gradiente)  # Ejemplo de uso
```

---

## ⚡ Compilación con Cython (opcional, recomendado para mayor velocidad)

### 1. Instala Cython y setuptools si no los tienes:

```bash
pip install cython setuptools
```

### 2. Compila el módulo Cython:

Desde la raíz del repositorio:

```bash
python setup.py build_ext --inplace
```

Esto generará el archivo compilado en `backtamal/engine_cython.*.so`.

### 3. Usa la versión compilada

Puedes importar la versión compilada igual que la versión pura de Python, solo cambiando el import:

```python
from backtamal.engine_cython import Valor as ValorCython
```

---

## 📝 Notas

- Si modificas el archivo `.pyx`, recuerda recompilar con el comando anterior.
- El código está pensado para aprendizaje, ¡experimenta y modifica a tu gusto!

---
