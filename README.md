# 🇵🇰 Pakistani Social Media Content Detective

> **Business question:** *What type of social-media content gets Pakistani audiences to engage?*

As a junior digital-marketing data scientist, I analysed 3,000 social-media posts to find which platforms, formats, topics, sentiments, posting times and audience segments drive engagement, then turned the patterns into an AI-assisted marketing recommendation.

## Questions answered

| # | Question | Finding |
|---|---|---|
| 1 | Which platform performs better? | **TikTok** (2.8% median engagement) - X/Twitter is lowest (1.06%) |
| 2 | Which content type gets more engagement? | **Short Video/Reel** (2.54%) - plain Text is lowest (0.95%) |
| 3 | Which topics/hashtags are popular? | **Eid & Ramadan** and **Cricket**; top tags: #Iftar, #GreenShirts, #ChandRaat, #BabarAzam |
| 4 | What sentiment generates engagement? | **Positive** captions (2.1%) beat Neutral (1.37%) |
| 5 | What time/day is better? | **Prime time 9-11 PM**, peak at 21:00; **Saturday** (Fri-Sun stronger than weekdays) |
| 6 | Which segment is most responsive? | **Urban Gen Z (18-24)**; Families (35+) lowest |
| 7 | What should a brand post next? | See recommendation below |

Engagement rate = (likes + comments + shares) / followers x 100. I use the **median** because a few viral posts would distort the mean.

## Recommendation for a Pakistani brand

Post a **positive, celebratory short video/Reel on TikTok** around an **Eid/Ramadan or cricket** theme, in a Roman Urdu + English mix, on a **Friday-Sunday evening around 9 PM**, targeting **urban Gen Z**. Use hashtags such as #Iftar, #ChandRaat, #GreenShirts.

AI-assisted step: `results/ai_prompt.txt` contains a ready-made prompt (with the findings embedded) to give to Claude/ChatGPT/Gemini to generate a 7-day content calendar, captions and A/B tests.

## Project structure

```
data/pakistan_social_media.csv   # dataset (3,000 posts)
generate_data.py                 # creates the dataset
analysis.py                      # cleaning, sentiment, analysis, charts, recommendation
charts/                          # 10 PNG charts
results/                         # ranking tables, key_findings.md, recommendation, ai_prompt.txt
```

## How to run

```bash
pip install -r requirements.txt
python generate_data.py     # optional - skip if you use your own data
python analysis.py
```

## Method

1. **Clean** - drop duplicates/missing values, compute engagement rate, add time slots.
2. **Sentiment** - lexicon-based classifier on captions (English + Roman Urdu words like *zabardast*, *bakwas*, *mehngai*).
3. **Analyse** - group by platform, content type, topic, hashtag, sentiment, time, day, segment; platform x content heatmap.
4. **Recommend** - take the top pattern from each dimension, then use an LLM prompt to expand it into a campaign.

## ⚠️ Limitations (important)

- The dataset is **simulated** (built with realistic Pakistani platforms, topics, cities and behaviour patterns) because real platform data is restricted by APIs. The findings therefore illustrate the *method*, not real-world truth.
- To use real data (Kaggle, Meta/YouTube exports, etc.), put a CSV with the same columns in `data/` and re-run `analysis.py`.
- Sentiment is a simple word-list approach; a transformer model (e.g. XLM-RoBERTa) would handle Urdu/Roman Urdu better.
- Correlation is not causation - validate with A/B tests.

## Tech stack
Python, pandas, NumPy, Matplotlib
