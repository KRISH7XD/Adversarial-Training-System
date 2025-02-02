
Adversarial Training System
🔒 Making AI Models Resilient Against Adversarial Attacks

📌 Overview
The Adversarial Training System is an innovative solution designed to strengthen AI models against adversarial attacks like FGM, PGD, CarliniL2, and DeepFool. These attacks exploit AI model vulnerabilities, leading to incorrect predictions and security risks. Our system enhances model robustness by retraining it with adversarial examples, making it more secure and reliable.

🚀 Features
✔ Automatic Model Security Enhancement – Upload a model, and the system will generate a more secure version.
✔ Supports Multiple Attack Types – Detects and protects against various adversarial attacks.
✔ Customizable Security Levels – Choose from different security levels (Fast, Low, Medium, High).
✔ Intuitive UI – A Streamlit-based interface for seamless interaction.
✔ Performance Visualization – Graphs showing accuracy and loss trends before and after training.

🛠Tech Stack
Programming Language: Python
Framework: Streamlit
Deep Learning: PyTorch
Adversarial AI: Adversarial Robustness Toolbox (ART)
Data Handling: NumPy, Pandas
Visualization: Matplotlib



📂 Project Structure

📂 Adversarial-Training-System  
 ├── 📂 uploads              # Folder to store uploaded models & datasets  
 ├── 📄 main.py              # Main script handling model training & UI  
 ├── 📄 requirements.txt     # List of dependencies  
 ├── 📄 README.md            # Project documentation  
 ├── 📄 secure_model.pth     # Example secured model output  
 
🔧 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-repo/adversarial-training-system.git
cd adversarial-training-system
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Application
streamlit run main.py

🎯 How It Works
1️⃣ Upload your pre-trained AI model (.pth/.pt) and dataset (CSV format).
2️⃣ Select the security level – Choose between Fast, Low, Medium, or High.
3️⃣ The system generates adversarial examples using attacks like PGD, CarliniL2, etc.
4️⃣ Retrains the model with adversarial and clean data to improve robustness.
5️⃣ Download the secured model, now immune to adversarial attacks.
6️⃣ View accuracy and loss trends via interactive plots.


📌 Why This Matters?
With AI being integrated into healthcare, finance, and autonomous systems, security is more crucial than ever. 
Adversarial attacks can manipulate AI decisions, leading to critical failures. This system ensures that AI models remain trustworthy and resilient against such threats.

🤝 Contributors
👤 Krish
👤 Aaryan Joshi
👤 Devansh Kapadia

🌟 Future Enhancements
✅ Extend support to different neural network architectures.
✅ Implement real-time attack detection for deployed AI models.
✅ Optimize training efficiency for faster adversarial training.

📜 License
This project is open-source under the MIT License.

⭐ Show Some Love!
If you find this project helpful, consider giving it a ⭐ on GitHub! 🚀
