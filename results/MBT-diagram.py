import graphviz

# Create a diagram of the formal MBT framework as described by Tretmans
dot = graphviz.Digraph(format='svg')
dot.attr(rankdir='LR', size='8,5')

# Nodes
dot.node('SPEC', 'SPEC\n(formal specifications)')
dot.node('MOD', 'MOD\n(models of implementations)')
dot.node('imp', 'imp ⊆ MOD × SPEC\n(correctness relation)')
dot.node('TEST', 'TEST\n(test cases)')
dot.node('passes', 'passes ⊆ MOD × TEST\n(test result)')
dot.node('gen_imp', 'gen_imp: SPEC → P(TEST)\n(test generation algorithm)')
dot.node('proof', 'Proof of soundness\n& exhaustiveness')

# Edges
dot.edge('SPEC', 'gen_imp', label='input to')
dot.edge('gen_imp', 'TEST', label='generates')
dot.edge('MOD', 'passes', label='run tests on')
dot.edge('TEST', 'passes', label='compare output')
dot.edge('MOD', 'imp', label='check conformance')
dot.edge('SPEC', 'imp', label='against spec')
dot.edge('imp', 'proof', label='supports')
dot.edge('passes', 'proof', label='supports')

dot.render('mbt_framework_diagram', cleanup=False)
dot.view('mbt_framework_diagram')

dot
