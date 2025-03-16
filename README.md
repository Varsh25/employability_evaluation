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

## 🛠️ **How It Works**  
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


