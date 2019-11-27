import networkx as nx 
import matplotlib.pyplot as plt 
import numpy as np 
import sys

#give each edge of a graph a random weight from 0 to 100
def generate_random_weights(G):
	for u,v in G.edges():
			G[u][v]['weight'] = np.random.randint(100)

#make and show a complete graph on 10 vertices
def make_and_show_graph():
	G = nx.complete_graph(10)
	generate_random_weights(G)
	pos = nx.spring_layout(G)
	weights = nx.get_edge_attributes(G,'weight')
	nx.draw(G,pos)
	nx.draw_networkx_labels(G, pos)
	nx.draw_networkx_edge_labels(G,pos,weights, with_labels=True)
	return G

#find the smallest distance to a vertex not yet visited
def find_smallest_distance(G, visited, dist):
	lowest_weight = -1
	lowest_weight_index = -1
	for n in G.nodes():
		if (lowest_weight == -1 or dist[n] < lowest_weight) and not visited[n]:
				lowest_weight = dist[n]
				lowest_weight_index = n
	return lowest_weight_index

#given a graph G and an int originalVertex describing the vertex to start at, returns a list
#containing the shortest distance from originalVertex to the given vertex
def dijkstra_dist(G,originalVertex):
	dist = list(G.nodes())
	visited = [False] * len(dist)
	vertex_queue = list(G.nodes())
	dist = [sys.maxsize] * len(G.nodes)
	dist[originalVertex] = 0
	for n in range(len(G.nodes)):
		nearest_neighbor = find_smallest_distance(G, visited, dist)
		visited[nearest_neighbor] = True
		for v in list(G.nodes()):
			if nearest_neighbor != v and G[nearest_neighbor][v]['weight'] > 0 and visited[v] == False:
					if dist[v] > dist[nearest_neighbor] + G[nearest_neighbor][v]['weight']:
							dist[v] = dist[nearest_neighbor] + G[nearest_neighbor][v]['weight']
	return dist

G = make_and_show_graph()

print(dijkstra_dist(G,0))
plt.show(G)
exit()