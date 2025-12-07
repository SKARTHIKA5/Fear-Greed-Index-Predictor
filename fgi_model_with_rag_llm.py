!pip install yfinance xgboost sentence-transformers faiss-cpu transformers accelerate pandas numpy matplotlib tqdm
!pip install mistral_inference

import yfinance as yf, pandas as pd, numpy as np, faiss, warnings, matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import xgboost as xgb
warnings.filterwarnings('ignore')

end = datetime.now(); start = end - timedelta(days=730)
spx = yf.download('^GSPC', start=start, end=end)
vix = yf.download('^VIX', start=start, end=end)

df = pd.concat([spx['Close'], vix['Close']], axis=1, join='inner').dropna()
df.columns = ['SPX', 'VIX']

df['Returns'] = df['SPX'].pct_change()
df['Volatility'] = df['Returns'].rolling(5).std()
df['FGI'] = 100*(1 - (df['Volatility']/df['Volatility'].max()))
for lag in [1,3,5]:
    df[f'Lag_{lag}'] = df['Returns'].shift(lag)
df = df.dropna()

X = df[['VIX','Returns','Lag_1','Lag_3','Lag_5']]
y = df['FGI']
split = int(len(df)*0.8)
X_train,X_test = X[:split],X[split:]
y_train,y_test = y[:split],y[split:]

model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
model.fit(X_train, y_train)
preds = model.predict(X_test)

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE:", rmse)
plt.plot(y_test.values,label='True'); plt.plot(preds,label='Pred'); plt.legend(); plt.show()

news = [
 "S&P500 falls amid inflation fears",
 "Tech stocks rally on strong earnings",
 "Investors cautious over geopolitical risks",
 "Fed signals possible rate cuts",
 "Market sentiment improves as volatility cools"
]
embedder = SentenceTransformer('all-MiniLM-L6-v2')
news_emb = embedder.encode(news, convert_to_numpy=True)
index = faiss.IndexFlatL2(news_emb.shape[1]); index.add(news_emb)

def retrieve_docs(query, k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D,I = index.search(q_emb,k)
    return [news[i] for i in I[0]]

from huggingface_hub import snapshot_download
from pathlib import Path
mistral_models_path = Path.home().joinpath('mistral_models', '7B-Instruct-v0.3')
mistral_models_path.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.3", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir=mistral_models_path)

from transformers import pipeline
llm = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3", device_map="auto", token="you_hugging_face api token")

def llm_reasoning_offline(numeric_forecast):
    docs = retrieve_docs("market sentiment and volatility today")
    context = "\n".join([f"[{i+1}] {d}" for i,d in enumerate(docs)])
    prompt = f"""
You are a financial analyst estimating the Fear & Greed Index (0-100).
Numeric model forecast: {numeric_forecast:.2f}
Evidence:
{context}
Give final FGI estimate (0-100) and a 3-sentence reasoning.
"""
    out = llm(prompt, max_new_tokens=120)
    return out[0]['generated_text']

import joblib
joblib.dump(model, 'xgboost_fgi_model.joblib')


