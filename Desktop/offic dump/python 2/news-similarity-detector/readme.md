# 📰 News Similarity Detector

A Python project to identify **similar or duplicate news articles** between *fake* and *real* datasets using **TF-IDF vectorization** and **cosine similarity**.  
This helps detect overlapping or plagiarized content, often useful for cleaning datasets or analyzing news redundancy.

---

## 📘 Overview

This project combines fake and real news datasets, transforms the text into numerical vectors using **TF-IDF**, and computes **cosine similarity** between every pair of articles.  
Articles with similarity above a certain threshold (default: `0.8`) are flagged as potential duplicates.

---

## ⚙️ Features

✅ Load and merge fake and real news datasets  
✅ Combine title and body for better context  
✅ Convert text to TF-IDF vectors (max 10,000 features)  
✅ Compute cosine similarity between all articles  
✅ Detect highly similar article pairs  
✅ Display top similar titles with similarity scores  

---
