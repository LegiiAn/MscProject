import json
import numpy as np
import matplotlib.pyplot as plt


def main():
    # 1. Charger les données
    try:
        with open("C:/Users/m_vit/Documents/MscProject/results/semantic_reasoning_results.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("[ERREUR] Le fichier semantic_reasoning_results.json est introuvable.")
        return

    # Configuration du style pour un rendu académique (LaTeX friendly)
    plt.style.use('bmh')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # ==========================================
    # Graphique 1 : Précision et Intervalles de Confiance
    # ==========================================
    ax1 = axes[0]
    
    cat_rate = data["category_specific"]["rate_pct"]
    cat_ci = data["category_specific"]["wilson_ci_95"]
    spat_rate = data["spatial"]["rate_pct"]
    spat_ci = data["spatial"]["wilson_ci_95"]
    
    # Calcul des erreurs relatives pour matplotlib (mean - lower, upper - mean)
    cat_err = [[cat_rate - cat_ci[0]], [cat_ci[1] - cat_rate]]
    spat_err = [[spat_rate - spat_ci[0]], [spat_ci[1] - spat_rate]]
    
    bars = ax1.bar(["Category-Specific\n(Object Name)", "Spatial\n(Left/Right)"], 
                   [cat_rate, spat_rate], 
                   yerr=[[cat_err[0][0], spat_err[0][0]], [cat_err[1][0], spat_err[1][0]]],
                   capsize=8, color=["#4C72B0", "#55A868"], alpha=0.9, edgecolor="black")
    
    ax1.axhline(50, color='red', linestyle='--', linewidth=2, label="Random Baseline (50%)")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Instruction Following Rate (%)", fontweight='bold')
    ax1.set_title("Instruction Following Accuracy\nwith 95% Confidence Intervals", fontweight='bold')
    ax1.legend()
    
    # Ajouter les valeurs textuelles sur les barres
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., 15, f"{height:.1f}%", 
                 ha='center', va='bottom', color='white', fontweight='bold', fontsize=12)

    # ==========================================
    # Graphique 2 : Comportement Bimodal (Histogramme des distances)
    # ==========================================
    ax2 = axes[1]
    
    distances = []
    for scene in data["per_scene"]:
        preds = scene.get("predictions", {})
        if "prompt_a" in preds and "prompt_b" in preds:
            pa = np.array(preds["prompt_a"])
            pb = np.array(preds["prompt_b"])
            dist = np.linalg.norm(pa - pb)
            distances.append(dist)
            
    # Définition des bins pour bien séparer l'immobilisme (0-10px) des vrais mouvements
    bins = [0, 10, 30, 60, 100, 150, 250]
    ax2.hist(distances, bins=bins, color="#C44E52", edgecolor="black", alpha=0.9)
    
    ax2.set_xlabel("Pixel Shift Between Prompts", fontweight='bold')
    ax2.set_ylabel("Number of Scenes", fontweight='bold')
    ax2.set_title("Prediction Shift Distribution\n(Proof of Bimodal Behavior)", fontweight='bold')
    ax2.set_xticks(bins)

    # ==========================================
    # Graphique 3 : Répartition des Échecs (Catastrophic Forgetting)
    # ==========================================
    ax3 = axes[2]
    
    # Extraction des volumes
    correct = data["category_specific"]["correct"]
    total_valid = data["category_specific"]["total"]
    ignored = total_valid - correct
    parse_fails = data["parse_failures"]
    
    sizes = [correct, ignored, parse_fails]
    labels = [f"Followed\n({correct})", f"Ignored / Blind\n({ignored})", f"Syntax Failures\n({parse_fails})"]
    colors = ["#4C72B0", "#E1A000", "#C44E52"]
    explode = (0, 0, 0.1) # Met en évidence les parse failures
    
    ax3.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', 
            shadow=False, startangle=140, textprops={'fontweight': 'bold'})
    ax3.set_title("Model Breakdown (Category Prompts)\nIncluding Generation Failures", fontweight='bold')

    # ==========================================
    # Sauvegarde
    # ==========================================
    plt.tight_layout()
    output_filename = "semantic_evaluation_graphs.pdf"
    plt.savefig(output_filename, format="pdf", dpi=300, bbox_inches="tight")
    print(f"[SUCCÈS] Graphiques générés et sauvegardés dans : {output_filename}")

if __name__ == "__main__":
    main()
