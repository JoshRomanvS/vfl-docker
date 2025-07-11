from graphviz import Digraph
import html  # for basic escaping

# ------------------------------------------------------------------ helpers
def esc(s: str) -> str:
    """
    Escape &, <, > and additionally [, ], {, } so Graphviz HTML labels accept them.
    """
    return (html.escape(s)                 # & < >
            .replace('[', '&#91;')
            .replace(']', '&#93;')
            .replace('{', '&#123;')
            .replace('}', '&#125;'))

def html_lbl(action: str, guard: str = '', update: str = '') -> str:
    """
    Build an HTML label with three rows: action (bold), guard, update.
    Guard in orange, update in green.
    """
    rows = f'<TR><TD ALIGN="LEFT"><B>{esc(action)}</B></TD></TR>'
    if guard:
        rows += f'<TR><TD ALIGN="LEFT"><FONT COLOR="#D84315">{esc(guard)}</FONT></TD></TR>'
    if update:
        rows += f'<TR><TD ALIGN="LEFT"><FONT COLOR="#2E7D32">{esc(update)}</FONT></TD></TR>'
    return f'<\n<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="2">\n{rows}\n</TABLE>\n>'


# ------------------------------------------------------------------ graph
dot = Digraph(format='png')
dot.graph_attr.update(rankdir='LR')
dot.node_attr.update(shape='ellipse', style='filled', fillcolor='#A5D6A7')
dot.edge_attr.update(fontname='Helvetica', fontsize='9')

# states
dot.node('Unlocked')
dot.node('Locked')
dot.node('Blocked')
dot.node('init', shape='point', width='0.15', style='')

# transitions
dot.edge('init', 'Unlocked',
         label=html_lbl('initialisation', update='attempts := 0'))

dot.edge('Unlocked', 'Locked')
dot.edge('Locked', 'Unlocked')


dot.edge('Locked', 'Locked',
         label=html_lbl('enter(pin)',
                        '[ pin ≠ "1234" & attempts < 3 ]',
                        '{ attempts := attempts + 1 }'))
dot.edge('Locked', 'Blocked',
         label=html_lbl('enter(pin)',
                        '[ pin ≠ "1234" & attempts ≥ 3 ]',
                        '{ blocked }'),
         color='#D32F2F', fontcolor='#D32F2F')


# ------------------------------------------------------------------ render
dot.render('smart_door_sts_pretty', view=True)
