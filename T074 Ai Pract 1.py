from collections import deque
#A) Implement the Breadth First Search algorithm to solve a given problem.
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F', 'G'],
    'D': ['B'],
    'E': ['B'],
    'F': ['C'],
    'G': ['C']
}

def bfs(start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        
        vertex = queue.popleft()
        print(vertex, end=" ")

        for neighbour in graph[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

print("BFS Traversal:")
bfs('A')

#B)  Implement the Iterative Depth First Search algorithm to solve the same problem.
graph = {
    1: [2, 3],
    2: [1, 4, 5],
    3: [1, 6],
    4: [2],
    5: [2, 7],
    6: [3],
    7: [5]
}

def iterative_dfs(start):
    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()

        if node not in visited:
            visited.add(node)
            print(node, end=" ")

            for neighbour in reversed(graph[node]):
                if neighbour not in visited:
                    stack.append(neighbour)

print("DFS Traversal:")
iterative_dfs(1)






