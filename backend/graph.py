#!/usr/bin/env python
"""
detailed_diagram_square.py

This script creates a detailed, square (1:1 aspect ratio) diagram of the backend data flow using Graphviz.
It now incorporates time data into the feature space via two alternative methods:
  1. Sinusoidal Time Encoding.
  2. Learned Time Embedding.
The diagram shows input JSON configurations, transformation steps, tensor shape changes (with equations),
and how the modules interconnect (including conversion from NetworkX graph to JSON for PyG).
The spacing between nodes has been reduced to minimize white space.
"""

from graphviz import Digraph

dot = Digraph(comment='Backend Data Flow Diagram', format='png')

# Global graph attributes for a square, compact layout.
dot.attr(rankdir='TB', size="8,8!", ratio="fill", splines='polyline', dpi="600")
dot.attr('graph', nodesep='0.2', ranksep='0.2', margin='0.05', concentrate='true')

# ===================== FEATURE SPACE CREATION (FeatureSpaceCreator) =========================
# Note: Input DataFrame now includes a "time" column along with "text" and "num"
dot.node('FSC_Input', 
         'Input DataFrame\nShape: (N, m)\nColumns: text, num, time', 
         shape='box', style='filled', fillcolor='lightblue')

# Process text as before
dot.node('FSC_TextPre', 
         'Text Preprocessing & Tokenization\n"raw text" → Tokens', 
         shape='box', style='filled', fillcolor='lightyellow')

dot.node('FSC_BERT', 
         'BERT Embedding\nTokens → Embedding\nShape: (768,)\nf: Text → ℝ⁷⁶⁸', 
         shape='box', style='filled', fillcolor='lightgreen')

dot.node('FSC_PCA', 
         'PCA Reduction\nℝ⁷⁶⁸ → ℝ^(target_dim)\n(e.g., 768 → 2)', 
         shape='box', style='filled', fillcolor='lightpink')

dot.node('FSC_TextOut', 
         'Output Text Embedding\nShape: (N, target_dim)', 
         shape='box', style='filled', fillcolor='lightcyan')

# Process numeric features as before
dot.node('FSC_Numeric', 
         'Numeric Processing\nStandardization\nInput: (N,1)', 
         shape='box', style='filled', fillcolor='lightgray')

dot.node('FSC_NumOut', 
         'Output Numeric Feature\nShape: (N, 1)', 
         shape='box', style='filled', fillcolor='khaki')

# New branch: Process time information
dot.node('FSC_TimePre', 
         'Extract "time" Column\nRaw time data', 
         shape='box', style='filled', fillcolor='thistle')

# Option 1: Sinusoidal Time Encoding
dot.node('FSC_TimeSin', 
         'Sinusoidal Time Encoding\nf(time) = [sin(ωt), cos(ωt), ...]\nShape: (N, time_dim)', 
         shape='box', style='filled', fillcolor='lightcoral')

# Option 2: Learned Time Embedding
dot.node('FSC_TimeLearn', 
         'Learned Time Embedding\nEmbedding Lookup\nShape: (N, time_dim)', 
         shape='box', style='filled', fillcolor='palegoldenrod')

# Decision node: Choose one method (or merge both)
dot.node('FSC_TimeOut', 
         'Output Time Feature\nShape: (N, time_dim)', 
         shape='box', style='filled', fillcolor='plum')

# Merge branches: Combine text, numeric, and time features
dot.node('FSC_Combined', 
         'Combined Feature Space\nColumns: text_embedding, num_feature, time_feature\nShape: (N, d_FSC)', 
         shape='box', style='filled', fillcolor='wheat')

# Edges for text and numeric as before
dot.edge('FSC_Input', 'FSC_TextPre', label='Extract "text"')
dot.edge('FSC_TextPre', 'FSC_BERT', label='Tokenized text')
dot.edge('FSC_BERT', 'FSC_PCA', label='BERT embedding (768-dim)')
dot.edge('FSC_PCA', 'FSC_TextOut', label='Reduce to target_dim')
dot.edge('FSC_Input', 'FSC_Numeric', label='Extract "num"')
dot.edge('FSC_Numeric', 'FSC_NumOut', label='Standardize')

# Edges for time branch
dot.edge('FSC_Input', 'FSC_TimePre', label='Extract "time"')
dot.edge('FSC_TimePre', 'FSC_TimeSin', label='Sinusoidal encoding', style='dotted')
dot.edge('FSC_TimePre', 'FSC_TimeLearn', label='Learned embedding', style='dotted')
# Use dashed edge to indicate alternative methods; choose one (or merge both)
dot.edge('FSC_TimeSin', 'FSC_TimeOut', label='Use Sinusoidal', style='dashed')
dot.edge('FSC_TimeLearn', 'FSC_TimeOut', label='Use Learned', style='dashed')

# Merge all features
dot.edge('FSC_TextOut', 'FSC_Combined', label='Add text_embedding')
dot.edge('FSC_NumOut', 'FSC_Combined', label='Add num_feature')
dot.edge('FSC_TimeOut', 'FSC_Combined', label='Add time_feature')

# ===================== DATAFRAME TO GRAPH (DataFrameToGraph) =========================
dot.node('DFG_Input', 
         'DataFrame Input\n(Includes: *_embedding, *_feature, time_feature)', 
         shape='box', style='filled', fillcolor='lightblue')

dot.node('DFG_Extract', 
         'Extract Node Features\nOutput: Feature dict for each node\nVector ∈ ℝ^(d)', 
         shape='box', style='filled', fillcolor='lightgreen')

dot.node('DFG_Graph', 
         'Construct NetworkX Graph\nNodes: (id, type, features)\nEdges: per relationship', 
         shape='box', style='filled', fillcolor='orange')

dot.edge('FSC_Combined', 'DFG_Input', label='Use processed DataFrame')
dot.edge('DFG_Input', 'DFG_Extract', label='Parse rows & extract features')
dot.edge('DFG_Extract', 'DFG_Graph', label='Create nodes & edges')

# ===================== TORCH GEOMETRIC GRAPH BUILDER (TorchGeometricGraphBuilder) =========================
dot.node('TGG_Input', 
         'JSON Node-Link Input\n{\n  "nodes": [{id, features, label}],\n  "links": [{source:{id}, target:{id}}]\n}', 
         shape='box', style='filled', fillcolor='lightblue')

dot.node('TGG_Mapping', 
         'Map Node IDs → Indices\nOutput: mapping (id → index)', 
         shape='box', style='filled', fillcolor='lightgreen')

dot.node('TGG_Features', 
         'Construct Feature Matrix x\nConcatenate features from nodes\nShape: (n, d)', 
         shape='box', style='filled', fillcolor='lightyellow')

dot.node('TGG_Structural', 
         'Add Structural Features\nx → [x, degree]\nShape: (n, d+1)', 
         shape='box', style='filled', fillcolor='lightpink')

dot.node('TGG_PCA', 
         'PCA Reduction\nℝ^(d+1) → ℝ^(target_dim_TGG)\n(e.g., (d+1) → 50)', 
         shape='box', style='filled', fillcolor='lavender')

dot.node('TGG_Split', 
         'Split Data\nCreate train/val/test masks\nFinal Data: x, edge_index, y, masks', 
         shape='box', style='filled', fillcolor='khaki')

dot.edge('TGG_Input', 'TGG_Mapping', label='Parse JSON')
dot.edge('TGG_Mapping', 'TGG_Features', label='Build x: (n, d)')
dot.edge('TGG_Features', 'TGG_Structural', label='Concatenate degree')
dot.edge('TGG_Structural', 'TGG_PCA', label='Apply PCA (optional)')
dot.edge('TGG_PCA', 'TGG_Split', label='Reduced x: (n, target_dim_TGG)')
dot.edge('TGG_Features', 'TGG_Split', label='If no PCA, use x', style='dashed')

# ===================== CONVERSION BETWEEN MODULES =====================
dot.node('Conversion', 
         'Conversion:\nNetworkX Graph → JSON Node-Link\n(Export from DFG)', 
         shape='ellipse', style='filled', fillcolor='yellow')
dot.edge('DFG_Graph', 'Conversion', label='Export graph as JSON')
dot.edge('Conversion', 'TGG_Input', label='Feed JSON to TGG')

# ===================== GRAPH LABEL (Concise) =====================
dot.attr(labelloc='b', label=('Params & Shapes:\n'
                               'FSC: BERT (768→target_dim, e.g.2); Standardize (N,1); Time: sin/cat learned → (N,time_dim); Combined: (N,target_dim+1(+time_dim))\n'
                               'DFG: Nodes: features ∈ ℝ^(d);\n'
                               'TGG: x: (n,d)→[x,degree]→(n,d+1)→PCA→(n,target_dim_TGG, e.g.50)'), 
         fontsize='12', fontcolor='black', fontname='Arial')

# Render the diagram into a PNG file.
dot.render('backend_data_flow_detailed_diagram_square', view=True)
print("Detailed square diagram generated as backend_data_flow_detailed_diagram_square.png")
