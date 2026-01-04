# llama-cpp-unity

A lightweight, no-nonsense wrapper for **llama.cpp** designed specifically for **Unity 6**.

This package provides a high-performance bridge to run Large Language Models (LLMs) locally within Unity using GGUF quantization. It is designed to be "just a wrapper"—giving you direct access to `llama.cpp` features without the overhead of a complex AI framework.

---

## ⚠️ Requirements

* **Unity 6** (6000.0.x) or higher.

---

## 🛠 Installation

To install **llama-cpp-unity** using the Unity Package Manager (UPM):

1. Open your project in **Unity 6**.
2. Navigate to **Window** > **Package Manager**.
3. Click the **+** (plus) icon in the top-left corner and select **Add package from git URL...**.
4. Enter the following URL: `https://github.com/lookbe/llama-cpp-unity.git`
5. Click **Add**.

---

## 🧪 Testing

1. Import the **BasicChat** sample project from the Package Manager.
2. Open the sample scene and locate the **ChatCompletion** component.
3. Change the **Model Path** to the **absolute path** of your model.

### Note on Mobile Paths (Android)
While you can extend the component to use `Application.streamingAssetsPath` on Desktop, **Android cannot load models directly from StreamingAssets**. You must use `Application.persistentDataPath` for the model path.

On Android, you must do one of the following:

* **Copying:** Copy the model from `StreamingAssets` to `persistentDataPath` first, before doing any model loading.
**OR**
* **Downloading:** Create a downloader script and save the model asset directly into `persistentDataPath`.

---
