from app.orchestration.hiring_graph import build_hiring_graph

# Build ONCE at import time
graph = build_hiring_graph()
