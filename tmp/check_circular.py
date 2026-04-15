import ast
import os
from collections import defaultdict

def get_module_name(file_path):
    path_parts = os.path.normpath(file_path).split(os.sep)
    if 'medicai' in path_parts:
        idx = path_parts.index('medicai')
        parts = path_parts[idx:]
        if parts[-1].endswith('.py'):
            parts[-1] = parts[-1][:-3]
        if parts[-1] == '__init__':
            parts = parts[:-1]
        return '.'.join(parts)
    return None

import_graph = defaultdict(set)
modules = {}

# Parse all modules in medicai.models.nnunet, medicai.trainer.nnunet, medicai.dataloader.nnunet
# Actually, just parse all of medicai
for root, _, files in os.walk('medicai'):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            mod_name = get_module_name(path)
            if not mod_name or 'nnunet' not in mod_name:
                continue
            
            with open(path, 'r', encoding='utf-8') as file:
                try:
                    tree = ast.parse(file.read())
                except SyntaxError:
                    continue
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        if 'medicai' in n.name and 'nnunet' in n.name:
                            import_graph[mod_name].add(n.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and 'medicai' in node.module and 'nnunet' in node.module:
                        import_graph[mod_name].add(node.module)

# DFS for cycles
def find_cycles(graph):
    visited = set()
    path = []
    cycles = []
    
    def dfs(node):
        if node in path:
            cycle_start = path.index(node)
            cycles.append(path[cycle_start:] + [node])
            return
        if node in visited:
            return
        
        visited.add(node)
        path.append(node)
        for neighbor in graph.get(node, []):
            dfs(neighbor)
        path.pop()

    for node in list(graph.keys()):
        dfs(node)
        
    return cycles

cyc = find_cycles(import_graph)
if cyc:
    for c in cyc:
        print("Cycle detected:", " -> ".join(c))
else:
    print("No circular imports detected!")
