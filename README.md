Regex Generator using Genetic Programming
🚀 Automatically infer regular expressions from examples using Genetic Programming (GP)

📌 Overview
This project uses Genetic Programming to evolve regular expressions based on positive and negative examples provided by the user. The best regex pattern is optimized for precision, recall, complexity, and diversity.

It is implemented using Streamlit for the user interface and DEAP (Distributed Evolutionary Algorithms in Python) for the genetic algorithm.

🎯 Features
🔹 User-friendly UI (Streamlit-based)

🔹 Multi-objective optimization using NSGA-II

🔹 Evolves regex patterns automatically

🔹 Uses crossover and mutation to refine regex

🔹 Displays the best regex at the end of evolution



Run the application : streamlit run ilast.py
📖 How It Works
1️⃣ Enter positive examples (text patterns you want to match).
2️⃣ Enter negative examples (text patterns to be avoided).
3️⃣ Click "Generate Regex" to start evolution.
4️⃣ The system runs 30 generations using Genetic Programming.
5️⃣ The best regex is displayed at the end.
