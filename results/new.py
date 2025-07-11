import graphviz

# Create Digraph
dot = graphviz.Digraph(format='svg')
dot.attr(rankdir='TB', size='8,6')

# --- Active Server ---
dot.node('SERVER', 'Active Server\n[Concatenation\nLogistic Regression\nCheckpoint]', shape='box')

# --- Passive Clients ---
for i in range(1, 4):
    label = f'Passive Client {i}\\n[Dense Layer\\nCheckpoint]'
    dot.node(f'CLIENT{i}', label, shape='box')

# --- Data flows: Intermediate results ---
dot.edge('CLIENT1', 'SERVER', label='Intermediate result')
dot.edge('CLIENT2', 'SERVER', label='Intermediate result')
dot.edge('CLIENT3', 'SERVER', label='Intermediate result')

# --- Data flows: Gradients ---
dot.edge('SERVER', 'CLIENT1', label='Gradients')
dot.edge('SERVER', 'CLIENT2', label='Gradients')
dot.edge('SERVER', 'CLIENT3', label='Gradients')

# Render and display
dot.render('vfl_prototype_architecture', cleanup=False)
dot.view('vfl_prototype_architecture')
