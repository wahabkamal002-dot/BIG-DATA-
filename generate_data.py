"""
Generates a SIMULATED Pakistani social-media dataset (3,000 posts).
Replace data/pakistan_social_media.csv with real scraped/exported data
(same column names) and analysis.py will work unchanged.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 3000

def pick(options_weights, n):
    keys = list(options_weights)
    p = np.array(list(options_weights.values()), dtype=float)
    return rng.choice(keys, size=n, p=p / p.sum())

platforms = {"Instagram": .27, "Facebook": .25, "TikTok": .22, "YouTube": .14, "X (Twitter)": .12}
platform_mult = {"TikTok": 1.5, "Instagram": 1.25, "YouTube": 1.0, "Facebook": .8, "X (Twitter)": .6}

ctypes = {"Short Video/Reel": .30, "Image": .22, "Carousel": .14, "Text": .14, "Poll": .08, "Live": .12}
ctype_mult = {"Short Video/Reel": 1.6, "Live": 1.3, "Carousel": 1.15, "Poll": 1.1, "Image": .9, "Text": .6}

topics = {"Cricket": .16, "Eid & Ramadan": .12, "Food": .16, "Fashion": .13,
          "Music & Drama": .14, "Tech": .10, "Education": .09, "Travel (Northern Areas)": .10}
topic_mult = {"Cricket": 1.5, "Eid & Ramadan": 1.45, "Food": 1.2, "Fashion": 1.1,
              "Music & Drama": 1.15, "Travel (Northern Areas)": 1.05, "Tech": .8, "Education": .7}
topic_tags = {
    "Cricket": ["#PSL", "#PakistanCricket", "#BabarAzam", "#GreenShirts"],
    "Eid & Ramadan": ["#EidMubarak", "#Ramadan2026", "#Iftar", "#ChandRaat"],
    "Food": ["#KarachiFood", "#DesiFood", "#BBQ", "#LahoriNashta"],
    "Fashion": ["#PakistaniFashion", "#LawnCollection", "#Shaadi", "#DesiStyle"],
    "Music & Drama": ["#CokeStudio", "#PakDrama", "#Lollywood", "#NewMusic"],
    "Tech": ["#PakTech", "#Freelancing", "#StartupPK", "#AI"],
    "Education": ["#StudyTips", "#MDCAT", "#ScholarshipPK", "#LearnWithMe"],
    "Travel (Northern Areas)": ["#Hunza", "#Skardu", "#NorthernPakistan", "#TravelPK"],
}

segments = {"Urban Gen Z (18-24)": .30, "Young Professionals (25-34)": .28,
            "Families (35+)": .22, "Small-City Youth (18-30)": .20}
seg_mult = {"Urban Gen Z (18-24)": 1.35, "Small-City Youth (18-30)": 1.15,
            "Young Professionals (25-34)": 1.0, "Families (35+)": .85}
seg_city = {
    "Urban Gen Z (18-24)": ["Karachi", "Lahore", "Islamabad"],
    "Young Professionals (25-34)": ["Karachi", "Lahore", "Islamabad", "Rawalpindi"],
    "Families (35+)": ["Karachi", "Lahore", "Faisalabad", "Multan"],
    "Small-City Youth (18-30)": ["Peshawar", "Quetta", "Sialkot", "Gujranwala", "Hyderabad"],
}

sentiments = {"Positive": .50, "Neutral": .30, "Negative": .20}
sent_mult = {"Positive": 1.25, "Negative": 1.0, "Neutral": .8}
languages = {"Roman Urdu": .40, "English": .35, "Urdu": .25}

caps = {
 "Positive": ["Absolutely loved this {t}! Zabardast!", "Best {t} moment ever, mazay aa gaye",
              "So proud of Pakistan, amazing {t} vibes", "Behtareen {t} content, love it"],
 "Negative": ["Very disappointed with {t}, bakwas experience", "Worst {t} ever, total waste",
              "Mehngai ne {t} ka sab kharab kar diya, so sad", "Poor quality, not happy with {t}"],
 "Neutral":  ["Here is today's update on {t}", "New {t} post is live, check the link",
              "{t} schedule for this week", "Sharing some {t} info"],
}

hour_mult = np.array([.6,.55,.5,.5,.55,.6,.7,.8,.85,.9,.95,1.0,1.05,1.05,1.0,1.0,1.05,1.1,1.15,1.25,1.35,1.4,1.3,.9])
day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
day_mult = {"Monday":.95,"Tuesday":.95,"Wednesday":1.0,"Thursday":1.05,"Friday":1.15,"Saturday":1.2,"Sunday":1.15}

df = pd.DataFrame({
    "platform": pick(platforms, N), "content_type": pick(ctypes, N), "topic": pick(topics, N),
    "audience_segment": pick(segments, N), "true_sentiment": pick(sentiments, N),
    "language": pick(languages, N),
})
df["city"] = [rng.choice(seg_city[s]) for s in df.audience_segment]
df["post_date"] = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, N), unit="D")
df["day_of_week"] = df.post_date.dt.day_name()
df["post_hour"] = rng.choice(24, size=N, p=hour_mult / hour_mult.sum())
df["hashtag"] = [rng.choice(topic_tags[t]) for t in df.topic]
df["caption"] = [rng.choice(caps[s]).format(t=t) + " " + h
                 for s, t, h in zip(df.true_sentiment, df.topic, df.hashtag)]
df["followers"] = np.round(rng.lognormal(10.2, 0.8, N)).astype(int).clip(2000, 500000)

mult = (df.platform.map(platform_mult) * df.content_type.map(ctype_mult) * df.topic.map(topic_mult)
        * df.audience_segment.map(seg_mult) * df.true_sentiment.map(sent_mult)
        * df.post_hour.map(lambda h: hour_mult[h] * 1.0) * df.day_of_week.map(day_mult))
rate = 0.012 * mult * rng.lognormal(0, 0.35, N)          # engagement as share of followers
inter = (df.followers * rate).round().astype(int).clip(lower=5)
df["likes"] = (inter * rng.uniform(.70, .85, N)).round().astype(int)
df["comments"] = (inter * rng.uniform(.05, .12, N)).round().astype(int)
df["shares"] = (inter * rng.uniform(.06, .15, N)).round().astype(int)
df["views"] = (df.followers * rng.uniform(1.5, 6, N) * df.content_type.map(
    lambda c: 1.8 if c in ("Short Video/Reel", "Live") else 1.0)).round().astype(int)

df.insert(0, "post_id", [f"P{i:04d}" for i in range(1, N + 1)])
df = df.drop(columns=["true_sentiment"])   # sentiment is re-derived from the caption in analysis.py
df.to_csv("data/pakistan_social_media.csv", index=False)
print(df.shape, "-> data/pakistan_social_media.csv")
