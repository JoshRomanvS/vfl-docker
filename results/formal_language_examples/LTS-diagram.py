import graphviz

# Create a Digraph object
dot = graphviz.Digraph(format='png')
dot.attr(rankdir='LR', size='10,5')
dot.attr('node', shape='circle')

# --- LTS A: Toggle Switch ---
# dot.node('s0', 's₀ (off)')
# dot.node('s1', 's₁ (on)')
# dot.edge('s0', 's1', label='?press')
# dot.edge('s1', 's0', label='?press')
# dot.edge('s1', 's1', label='!on')
# dot.edge('s0', 's0', label='!off')

# # --- LTS B: Vending Machine ---
# dot.node('v0', 'v₀ (idle)')
# dot.node('v1', 'v₁ (coin in)')
# dot.node('v2', 'v₂ (chosen)')
# dot.edge('v0', 'v1', label='?coin')
# dot.edge('v1', 'v2', label='?drink')
# dot.edge('v2', 'v0', label='!serve')

# # --- LTS C: Phone Call ---
dot.node('p0', 'p₀ (idle)')
dot.node('p1', 'p₁ (ringing)')
dot.node('p2', 'p₂ (in call)')
dot.edge('p0', 'p1', label='?call')
dot.edge('p1', 'p1', label='!ring')
dot.edge('p1', 'p2', label='?answer')
dot.edge('p2', 'p2', label='!connect')
dot.edge('p2', 'p0', label='?end')

# Render and display the diagram
output_path = "lts_diagrams"
dot.render(filename=output_path, format='png', cleanup=False)

dot.view(output_path)
