"""
Pakistani Social Media Content Detective - analysis pipeline.
Run: python generate_data.py && python analysis.py
"""
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
GREEN = "#01411C"

# ---------- 1. Load & clean ----------
df = pd.read_csv("data/pakistan_social_media.csv", parse_dates=["post_date"])
df = df.drop_duplicates("post_id").dropna(subset=["followers", "likes", "comments", "shares"])
df = df[df.followers > 0]
df["interactions"] = df.likes + df.comments + df.shares
df["engagement_rate"] = df.interactions / df.followers * 100          # % of followers
df["is_weekend"] = df.day_of_week.isin(["Friday", "Saturday", "Sunday"])  # Pakistani weekend-leaning days
bins = [-1, 5, 11, 16, 20, 23]
labels = ["Late night (0-5)", "Morning (6-11)", "Afternoon (12-16)", "Evening (17-20)", "Prime time (21-23)"]
df["time_slot"] = pd.cut(df.post_hour, bins=bins, labels=labels)

# ---------- 2. Sentiment (lexicon-based, English + Roman Urdu) ----------
POS = {"loved", "love", "best", "proud", "amazing", "zabardast", "behtareen", "mazay", "great", "awesome"}
NEG = {"disappointed", "bakwas", "worst", "waste", "sad", "poor", "kharab", "mehngai", "terrible"}

def sentiment(text):
    tokens = re.findall(r"[a-z']+", text.lower())
    score = sum(t in POS for t in tokens) - sum(t in NEG for t in tokens)
    if "not happy" in text.lower():
        score -= 1
    return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"

df["sentiment"] = df.caption.apply(sentiment)

# ---------- 3. Helpers ----------
def rank(col, data=df):
    return (data.groupby(col, observed=True).engagement_rate
            .agg(posts="count", median_er="median", mean_er="mean")
            .sort_values("median_er", ascending=False).round(2))

def bar(table, title, fname, horizontal=True):
    t = table.sort_values("median_er")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    (ax.barh if horizontal else ax.bar)(t.index.astype(str), t.median_er, color=GREEN)
    ax.set_title(title, fontweight="bold")
    (ax.set_xlabel if horizontal else ax.set_ylabel)("Median engagement rate (% of followers)")
    for i, v in enumerate(t.median_er):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8) if horizontal else None
    plt.tight_layout(); plt.savefig(f"charts/{fname}"); plt.close()

# ---------- 4. Answer the 7 business questions ----------
plat, ctype, topic = rank("platform"), rank("content_type"), rank("topic")
sent, slot, day, seg = rank("sentiment"), rank("time_slot"), rank("day_of_week"), rank("audience_segment")
tags = rank("hashtag").head(10)
hour = rank("post_hour").sort_index()
combo = (df.groupby(["platform", "content_type"]).engagement_rate.median().unstack().round(2))

bar(plat, "Q1. Which platform performs better?", "1_platform.png")
bar(ctype, "Q2. Which content type gets more engagement?", "2_content_type.png")
bar(topic, "Q3a. Which topics are most engaging?", "3a_topics.png")
bar(tags, "Q3b. Top 10 hashtags by engagement", "3b_hashtags.png")
bar(sent, "Q4. Which sentiment drives engagement?", "4_sentiment.png")
bar(slot, "Q5a. Best time of day", "5a_time_slot.png")
d = day.reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).sort_index(ascending=False)
bar(d.sort_values("median_er"), "Q5b. Best day of the week", "5b_day.png")
bar(seg, "Q6. Which audience segment is most responsive?", "6_segment.png")

fig, ax = plt.subplots(figsize=(8, 4.5))
im = ax.imshow(combo.values, cmap="Greens", aspect="auto")
ax.set_xticks(range(combo.shape[1])); ax.set_xticklabels(combo.columns, rotation=30, ha="right")
ax.set_yticks(range(combo.shape[0])); ax.set_yticklabels(combo.index)
for i in range(combo.shape[0]):
    for j in range(combo.shape[1]):
        ax.text(j, i, f"{combo.values[i, j]:.1f}", ha="center", va="center", fontsize=8)
ax.set_title("Platform x Content type (median engagement %)", fontweight="bold")
plt.colorbar(im); plt.tight_layout(); plt.savefig("charts/7_platform_x_content_heatmap.png"); plt.close()

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(hour.index, hour.median_er, marker="o", color=GREEN)
ax.set_title("Engagement by posting hour", fontweight="bold")
ax.set_xlabel("Hour of day (24h)"); ax.set_ylabel("Median engagement %"); ax.set_xticks(range(0, 24, 2))
plt.tight_layout(); plt.savefig("charts/5c_hourly.png"); plt.close()

# ---------- 5. Pattern -> recommendation ----------
best = {
    "platform": plat.index[0], "content": ctype.index[0], "topic": topic.index[0],
    "sentiment": sent.index[0], "slot": slot.index[0], "day": day.index[0],
    "segment": seg.index[0], "hashtags": ", ".join(tags.index[:3]),
    "hour": int(hour.median_er.idxmax()),
}
lift = lambda t: t.median_er.iloc[0] / t.median_er.iloc[-1]

for name, t in [("platform", plat), ("content_type", ctype), ("topic", topic), ("sentiment", sent),
                ("time_slot", slot), ("day_of_week", day), ("audience_segment", seg), ("hashtag", tags)]:
    t.to_csv(f"results/{name}_ranking.csv")
combo.to_csv("results/platform_x_content.csv")

summary = f"""# Key Findings (auto-generated)

Dataset: {len(df):,} posts | overall median engagement rate: {df.engagement_rate.median():.2f}% of followers

| Question | Winner | Median ER | Weakest | Gap |
|---|---|---|---|---|
| Platform | **{plat.index[0]}** | {plat.median_er.iloc[0]}% | {plat.index[-1]} ({plat.median_er.iloc[-1]}%) | {lift(plat):.1f}x |
| Content type | **{ctype.index[0]}** | {ctype.median_er.iloc[0]}% | {ctype.index[-1]} ({ctype.median_er.iloc[-1]}%) | {lift(ctype):.1f}x |
| Topic | **{topic.index[0]}** | {topic.median_er.iloc[0]}% | {topic.index[-1]} ({topic.median_er.iloc[-1]}%) | {lift(topic):.1f}x |
| Sentiment | **{sent.index[0]}** | {sent.median_er.iloc[0]}% | {sent.index[-1]} ({sent.median_er.iloc[-1]}%) | {lift(sent):.1f}x |
| Time slot | **{slot.index[0]}** | {slot.median_er.iloc[0]}% | {slot.index[-1]} ({slot.median_er.iloc[-1]}%) | {lift(slot):.1f}x |
| Day | **{day.index[0]}** | {day.median_er.iloc[0]}% | {day.index[-1]} ({day.median_er.iloc[-1]}%) | {lift(day):.1f}x |
| Audience segment | **{seg.index[0]}** | {seg.median_er.iloc[0]}% | {seg.index[-1]} ({seg.median_er.iloc[-1]}%) | {lift(seg):.1f}x |

Top hashtags: {', '.join(tags.index[:5])}
Peak posting hour: {best['hour']}:00
"""

recommendation = f"""# Data-Driven Marketing Recommendation (rule-based draft)

**Next post for a Pakistani brand**
- **Platform:** {best['platform']}
- **Format:** {best['content']}
- **Topic angle:** {best['topic']} (hashtags: {best['hashtags']})
- **Tone:** {best['sentiment']} - celebratory, proud, Roman Urdu/English mix
- **When:** {best['day']}, around {best['hour']}:00 ({best['slot']})
- **Target segment:** {best['segment']}

**Why:** each choice above ranked #1 by median engagement rate in this dataset.
**Caution:** correlation is not causation; test with an A/B campaign before scaling spend.
"""
open("results/key_findings.md", "w", encoding="utf-8").write(summary)
open("results/recommendation_rule_based.md", "w", encoding="utf-8").write(recommendation)

# ---------- 6. AI-assisted step: prompt to give any LLM (Claude/ChatGPT/Gemini) ----------
prompt = f"""You are a senior digital-marketing strategist for the Pakistani market.
Below are findings from analysing {len(df):,} social-media posts (engagement rate = interactions / followers).

{summary}

Platform x content-type engagement matrix:
{combo.to_string()}

Task:
1. Write a 7-day content calendar for a Pakistani food/fashion brand using these patterns.
2. Write 3 sample captions (Roman Urdu + English) with hashtags for the top topic.
3. Suggest 2 A/B tests to validate the findings.
4. List 2 risks or limitations of this data.
Keep it practical and culturally aware (Eid, PSL, weekends).
"""
open("results/ai_prompt.txt", "w", encoding="utf-8").write(prompt)
print(summary); print(recommendation)
