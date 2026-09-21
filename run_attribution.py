import json, warnings; warnings.filterwarnings("ignore")
from quant.attribution import run_attribution
from quant.agents.decision_tree import DecisionTreeAgent
from quant.agents.random_forest import RandomForestAgent
from quant.agents.gradient_boost import GradientBoostAgent
from quant.agents.lasso_factors import LassoFactorAgent
from quant.agents.rates_macro import RatesMacroAgent

agents = [DecisionTreeAgent(), RandomForestAgent(n_estimators=100),
          GradientBoostAgent(max_iter=120), LassoFactorAgent(), RatesMacroAgent()]
instruments = ["EURUSD","GBPUSD","XAUUSD","USOIL","ES1!","NQ1!"]

df, per_agent = run_attribution(agents, instruments, horizon=5, warmup=700, step=20)
df.to_csv("attribution_results.csv", index=False)

print("=== PER (agent, instrument) ===")
print(df[["agent","instrument","n","n_directional","hit_rate","IC"]].to_string(index=False))
print("\n=== PER AGENT (aggregate across instruments) ===")
print(json.dumps(per_agent, indent=2))
with open("attribution_summary.json","w") as f:
    json.dump(per_agent, f, indent=2)
print("\nDONE")
