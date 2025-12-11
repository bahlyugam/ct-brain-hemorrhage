import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Reload and preprocess
file_path = "data/brain_ct_pathlogy_characterization.xlsx"
df = pd.read_excel(file_path, sheet_name='Pathology characterization')
df['Pathology'] = df['Pathology'].astype(str).str.strip()

# Filter out very long non‑pathology rows
df = df[df['Pathology'].str.len() < 50].copy()

# --- Mapping rules to collapse similar pathologies --- #
def map_category(path: str) -> str:
    p = path.lower()
    if 'atrophy' in p:
        return 'Diffuse Cerebral Atrophy'
    if 'hydrocephalus' in p or ('ventricle' in p and 'enlarge' in p):
        return 'Hydrocephalus'
    if 'hemorr' in p or 'hematoma' in p:
        return 'Intracranial Hemorrhage'
    if 'infarct' in p or 'ischem' in p or 'small vessel' in p or 'lacunar' in p:
        return 'Ischemic Injury / Infarct'
    if 'cyst' in p:
        return 'Cystic Lesion'
    if any(k in p for k in ['tumor', 'glioma', 'meningioma', 'neoplasm', 'metastasis', 'mass', 'lymphoma', 'cytoma', 'sol']):
        return 'Neoplasm / Mass'
    if any(k in p for k in ['aneurysm', 'avm', 'vascular', 'loop', 'thrombosis']):
        return 'Vascular Lesion'
    if any(k in p for k in ['sinus', 'mucosal', 'rhinitis', 'polyp']):
        return 'Paranasal Sinus Disease'
    if any(k in p for k in ['fracture', 'injury', 'contusion', 'trauma']):
        return 'Trauma / Skull Injury'
    if 'edema' in p:
        return 'Cerebral Edema'
    if any(k in p for k in ['calcific', 'calcified', 'calcification', 'granuloma']):
        return 'Calcification / Granuloma'
    if any(k in p for k in ['encephalitis', 'abscess', 'mening', 'inflamm']):
        return 'Infection / Inflammation'
    if 'demyelin' in p:
        return 'Demyelinating Disease'
    if 'artifact' in p or 'variant' in p or 'normal' in p or 'suboptimal' in p:
        return 'Normal Variant / Artifact'
    return 'Other'

df['Combined'] = df['Pathology'].apply(map_category)

# --- Build condensed table --- #
condensed = (
    df.groupby('Combined')
      .agg(
          Count=('Pathology', 'nunique'),
          Constituent_Pathologies=('Pathology', lambda x: ', '.join(sorted(set(x))[:6])),
          Example_Meaning=('Meaning in Plain Language', 'first')
      )
      .reset_index()
      .sort_values('Combined')
)

# --- Build knowledge graph --- #
G = nx.DiGraph()

# Add nodes
for cat in condensed['Combined']:
    G.add_node(cat)

# Add edges based on co‑occurrence & confusion (mapped to combined cats)
for _, row in df.iterrows():
    src = row['Combined']
    # Co‑occurring
    if pd.notnull(row['Co-occurring Pathologies']) and row['Co-occurring Pathologies'] != 'None specific':
        for item in [c.strip() for c in str(row['Co-occurring Pathologies']).split(',')]:
            if item:
                tgt = map_category(item)
                G.add_edge(src, tgt, relation='co‑occurs with')
    # Confusions
    if pd.notnull(row['Likely Confused With']) and row['Likely Confused With'] != 'None specific':
        for item in [c.strip() for c in str(row['Likely Confused With']).split(',')]:
            if item:
                tgt = map_category(item)
                G.add_edge(src, tgt, relation='confused with')

# Plot
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, k=1.2, iterations=100, seed=42)
edge_colors = ['gold' if G[u][v]['relation'] == 'co‑occurs with' else 'crimson' for u, v in G.edges]

nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1200, alpha=0.9)
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrowstyle='-|>', arrowsize=15, width=2, alpha=0.7)
nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

# Legend
import matplotlib.patches as mpatches
legend_elements = [
    mpatches.Patch(color='gold', label='co‑occurs with'),
    mpatches.Patch(color='crimson', label='confused with')
]
plt.legend(handles=legend_elements, loc='upper left')
plt.title("Knowledge Graph of Common Brain CT Pathologies (Grouped)", fontsize=14)
plt.axis('off')
plt.show()
