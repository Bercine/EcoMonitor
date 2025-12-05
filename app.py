from flask import Flask, send_file, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
df = pd.read_csv("data.csv")

# Page principale
@app.route("/")
def dashboard():
    return send_file("dashboard.html")

# Statistiques globales
@app.route("/api/stats")
def stats():
    return jsonify({
        "energie_totale": float(df["Energie_kWh"].sum()),
        "cout_total": float(df["Cout_licence_euros"].sum()),
        "score_nird_moyen": float(df["Score_NIRD"].mean()),
        "nb_postes_total": int(df["Nb_postes"].sum())
    })

# Graphiques par salle
@app.route("/api/energie_par_salle")
def energie_par_salle():
    return jsonify(df.groupby("Salle")["Energie_kWh"].sum().to_dict())

@app.route("/api/cout_par_salle")
def cout_par_salle():
    return jsonify(df.groupby("Salle")["Cout_licence_euros"].sum().to_dict())

@app.route("/api/nird_par_salle")
def nird_par_salle():
    return jsonify(df.groupby("Salle")["Score_NIRD"].mean().to_dict())

@app.route("/api/nb_postes")
def nb_postes():
    return jsonify(df.groupby("Salle")["Nb_postes"].sum().to_dict())

# Recommandations : licences coûteuses
@app.route("/api/reco_licences")
def reco_licences():
    data = df[df["Licence"]=="Propriétaire"].set_index("Logiciel")["Cout_licence_euros"].to_dict()
    return jsonify(data)

# Recommandations : ordinateurs fin de vie
@app.route("/api/reco_finvie")
def reco_finvie():
    data = df[df["Duree_vie_annees"] <= 1].groupby("Salle").size().to_dict()
    return jsonify(data)

# Recommandations : score NIRD faible
@app.route("/api/reco_nird")
def reco_nird():
    data = df[df["Score_NIRD"] < 50].groupby("Salle").size().to_dict()
    return jsonify(data)

# Distribution des OS
@app.route("/api/os_distribution")
def os_distribution():
    return jsonify(df.groupby("OS").size().to_dict())

# Usage vs Energie
@app.route("/api/usage_vs_energie")
def usage_vs_energie():
    return jsonify(df[["Usage_hebdo_h","Energie_kWh"]].to_dict(orient="list"))

# Recyclable
@app.route("/api/recyclable")
def recyclable():
    data = df.groupby("Recyclable").size().to_dict()
    return jsonify(data)

# Prédictions futures
@app.route("/api/predictions")
def predictions():
    X = df[["Usage_hebdo_h"]].values
    y = df["Energie_kWh"].values
    model = LinearRegression().fit(X, y)
    pred_energy = float(model.predict(np.array([[X.mean()]])))
    future_cost = df.apply(lambda row: row["Cout_licence_euros"]*1.1 if row["Licence"]=="Propriétaire" else 0, axis=1).sum()
    return jsonify({
        "energie_futur": round(pred_energy,2),
        "cout_futur": round(future_cost,2)
    })

if __name__ == "__main__":
    app.run(debug=True)
