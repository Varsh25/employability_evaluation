# 🎓 **AI-Powered Employability Evaluation Tool** 🚀  

A **Deep Learning & Perceptron-based AI model** that assesses **employability skills** based on various factors such as **communication, confidence, presentation skills, and performance ratings**.  
 
📊 **Dataset Used:** [**Students Employability Dataset**](https://www.kaggle.com/code/sayakghosh001/students-employability-dataset/input)  

---

## 🚀 **Project Overview**  
This project evaluates **job readiness** using **AI-powered assessment models**. The tool takes various inputs (e.g., communication skills, mental alertness, confidence) and predicts whether a candidate is **Employable ✅ or LessEmployable ❌**.  

### **🔹 Models Used:**  
✔ **Perceptron (Baseline Model)** → A simple neural network for classification.  
✔ **Deep Learning Model (ANN)** → A more advanced artificial neural network for better accuracy.  

### 🛠 **Built with:**  
✅ **Scikit-learn** (Perceptron Model)  
✅ **TensorFlow/Keras** (Deep Learning Model)  
✅ **Gradio** (Interactive Web UI)  
✅ **Hugging Face Spaces** (Hosting & Deployment)  

---

## 🤜 **How It Works**  
1️⃣ **User enters ratings (1-5) for different employability factors**.  
2️⃣ **Inputs are preprocessed & scaled**.  
3️⃣ **Trained AI models (Perceptron & Deep Learning) predict** whether the candidate is **Employable or LessEmployable**.  
4️⃣ **Displays confidence scores** to indicate AI’s prediction accuracy.  

---

## 📌 **Example Output**  

| **Input Factors (1-5)**                                   | **Perceptron Prediction** | **Deep Learning Prediction** | **Confidence Score** |
|-----------------------------------------------------------|--------------------------|------------------------------|----------------------|
| Appearance: 4, Speaking: 5, Confidence: 5, Communication: 5  | **Employable ✅**          | **Employable ✅**              | **98.5%**            |
| Appearance: 3, Speaking: 3, Confidence: 2, Communication: 3  | **LessEmployable ❌**      | **LessEmployable ❌**          | **92.3%**            |

---

### **Install Dependencies**  
```bash
pip install gradio scikit-learn tensorflow numpy pandas
```

---

## 📎 **Files in This Repository**  

| 📁 **Folder** | 📄 **File Name**             | 🌟 **Description** |
|--------------|-----------------------------|---------------------|
| **Root**     | `train_model.py`             | **Script for training Perceptron & Deep Learning models** |
| **Root**     | `app.py`                     | **Main script for running the Gradio app** |
| **Root**     | `perceptron_model.pkl`       | **Trained Perceptron model** |
| **Root**     | `deep_learning_model.h5`     | **Trained Deep Learning (ANN) model** |
| **Root**     | `scaler.pkl`                 | **StandardScaler for feature scaling** |
| **Root**     | `requirements.txt`           | **Dependencies needed for deployment** |
| **Root**     | `README.md`                  | **Project documentation** |

---

## 🚀 **Deployment on Hugging Face**  
This project is hosted on **Hugging Face Spaces** using **Gradio** for easy access.  

### **📌 Steps to Deploy:**  
**1️⃣ Train the models** in Google Colab using `train_model.py`.  
**2️⃣ Download the saved models** (`perceptron_model.pkl`, `deep_learning_model.h5`, `scaler.pkl`).  
**3️⃣ Create a new Hugging Face Space** → [Click Here](https://huggingface.co/spaces).  
**4️⃣ Upload the following files:**  
   - `app.py`  
   - `perceptron_model.pkl`  
   - `deep_learning_model.h5`  
   - `scaler.pkl`  
   - `requirements.txt`  
**5️⃣ Restart the Hugging Face Space** and **share the live link!** 🎉  

🔗 **Try it live on Hugging Face:** [Click Here](https://huggingface.co/spaces/srivarshini25/employability_evaluation)  

---

## 📊 **Model Performance**  
✔ **Perceptron & Deep Learning models trained on employability skills dataset**.  
✔ **High accuracy in predicting employability**.  
✔ **Fast, real-time predictions** with confidence scores.  

### **🛡️ Accuracy Comparison**  
| **Model**           | **Accuracy**  |
|--------------------|-------------|
| **Perceptron**     | **85.6%**    |
| **Deep Learning**  | **92.4%**    |

---

## 📉 **Future Enhancements**  
🔹 **Enhance accuracy** by using a more complex **Deep Learning model (LSTMs, BERT)**.  
🔹 **Add Resume Screening** to analyze CVs and provide a **detailed job readiness score**.  
🔹 **Integrate Speech & Facial Analysis** for advanced assessments.  

---

## 🤝 **About Me**  
👋 Hi, I'm **Srivarshini R**, the creator of this project!  
This is part of my **learning journey in AI & Deep Learning**.  
If you have feedback or suggestions, feel free to reach out!  

📩 **Connect with me:**  
- **GitHub:** [https://github.com/Varsh25?tab=repositories]  
- **LinkedIn:** [https://www.linkedin.com/in/srivarshini-ramakrishnan/]  
- **Hugging Face:** [https://huggingface.co/srivarshini25]  

---

**Hope you find this project useful!** If you like it, give it a ⭐ and share your feedback. 🚀😊  

