# Datascience-and-Advanced-Programming-2025-2026-project-JK
Project Title:
Predicting Political Stability Using Economic and Social Indicators
Category:
Data Analysis & Machine Learning (Geopolitics / Economics)

Problem Statement or Motivation
Political stability is fundamental for a country — especially for its leadership, investment climate and the welfare of its citizens. Recently, we have witnessed major protests and regime change in countries such as Madagascar in late 2025, where youth-led demonstrations and a military takeover followed widespread dissatisfaction over basic services and governance. Similarly, in Nepal in early September 2025, mass protests brought down the government amid frustration with corruption, social media bans and elite capture.These events highlight how quickly stability can be eroded. This trend could see Tanzania’s leaders follow the same road very soon. The core problem of this project is: This project focuses on predicting the stability of a country: can we forecast a country’s stability one year ahead from observable economic and social indicators? The goal is to build and evaluate supervised ML models that predict the Fragile States Index (FSI) out of sample . This question motivates me deeply: being passionate about geopolitics, I am fascinated by how major changes in a country’s political situation may be explained, or even anticipated, through social/economic data trends.

Planned Approach and Technologies
I will draw on open international datasets:
* Fund for Peace’s Fragile States Index (FSI) will serve as the target variable — a yearly stability score per country.
* World Bank Open Data will provide features such as GDP per capita, unemployment, trade openness, inflation and public debt.
* United Nations Development Programme (UNDP) Human Development Index will supply education, health and income components (social variables). The workflow will begin with data acquisition and cleaning.Then, I will code and use as a benchmark model using panel regression with fixed effect. Next I will train and compare various machine learning models (2 supervised,Random Forest, XGBoost, 1 semi-supervised, Label Spreading, and 1 unsupervised, K-means clustering ) to estimate each country’s stability score and compare them fo find best one.( As a stretch goal I may also include Monte Carlo simulations to explore causal impacts of major shocks and scenario-based future instability risk and see if it affects its stability score.)

Expected Challenges and How I’ll Address Them
* Country selection and coverage: I may not use all countries if data quality varies. I will select a subset of countries with reliable data across years to ensure consistency.
* Data quality and missing values: Different sources may have gaps ,misalignments or the trend is not explaining what it should be.
* Model choice and interpretability: Choosing the right machine learning model is crucial, ill have to do some more research on the different machine learning model and on the hyperparameters.

Success Criteria
* Machine learning accuracy >-70-% .
* Clear visualisations ccomapring model stability score and real one.
* A reproducible codebase with documentation and notebooks that support the project’s findings.

Stretch Goals
* Develop an interactive dashboard (Streamlit) where users explore predicted stability by country and scenario with live statistique. ( probably unrealisable)
* Extend the model to predict future probability of instability (e.g., for 2026-2030) using Monte Carlo scenario simulations.( better for my goals and could happen if time permits)
