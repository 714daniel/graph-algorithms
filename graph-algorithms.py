import networkx as nx 
import matplotlib.pyplot as plt 
import numpy as np 
import sys

GRAPH_SIZE = 20
MAX_WEIGHT = 100

#get weight of edge
def getWeight(e):
	return G[e[0]][e[1]]['weight']

#give each edge of a graph a random weight from 0 to MAX_WEIGHT
def generate_random_weights(G):
	for u,v in G.edges():
		G[u][v]['weight'] = np.random.randint(MAX_WEIGHT)


#generate random weights, making sure graph satisfies triangle inequality such that w(AB) + w(BC) > w(AC)
def generate_triangle_inequality_weight(G):
	for u,v in G.edges():
		G[u][v]['weight'] = -1
	for u in G.nodes():
		for v in G[u]:
			for w in G[v]:
					if w != u:
						if G[u][v]['weight'] < 0:
							G[u][v]['weight']  = np.random.randint(1,MAX_WEIGHT)
						if (G[u][w]['weight'] > 0 and G[v][w]['weight'] > 0):
							G[u][w]['weight'] = np.random.randint(1,G[u][v]['weight'] + G[v][w]['weight']) #cap weight of ac at ab+bc

#find solution to TSP using nearest neighbor algorithm, starting from vertex v
def nearest_neighbor(G,v):
	visited = [False] * len(G.nodes)
	visited[v] = True
	curVertex = v
	curSum = 0
	path = list()

	while False in visited:
		lowest_weight = -1
		lowest_weight_vertex = -1
		for w in G[curVertex]:
			if visited[w] == False:
				if lowest_weight == -1 or G[curVertex][w]['weight'] < lowest_weight:
					lowest_weight_vertex = w
					lowest_weight = G[curVertex][w]['weight']
		curVertex = lowest_weight_vertex
		visited[lowest_weight_vertex] = True
		curSum += lowest_weight
		path.append(lowest_weight_vertex)
	return [path, curSum]

#produce minimum spanning tree as Graph obj
def find_mst_with_kruskal(G):
	sorted_edges = list(G.edges())
	sorted_edges.sort(key = getWeight)
	mst = nx.Graph()
	vertices = set()
	all_vertices = set(G.nodes())
	edge_index = 0
	while vertices != all_vertices or not nx.is_connected(mst):
		curEdge = sorted_edges[edge_index]
		if not curEdge[0] in vertices or not curEdge[1] in vertices or not nx.has_path(mst,curEdge[0],curEdge[1]):
			if curEdge[0] not in vertices:
				vertices.add(curEdge[0])
				mst.add_node(curEdge[0])
			if curEdge[1] not in vertices:
				vertices.add(curEdge[1])
				mst.add_node(curEdge[1])		
			mst.add_edge(*curEdge)
			mst[curEdge[0]][curEdge[1]]['weight'] = getWeight(curEdge)
		edge_index += 1
	return mst

#find approximate path between two vertices using mst
def approx_path(G,u,v):
	mst = find_mst_with_kruskal(G)
	path = nx.all_simple_paths(mst,u,v) #if there is more than 1, we messed up since we should have mst, so just use the first
	sum = 0
	for p in path:
		for i in range(len(p) - 1):
			sum += mst[p[i]][p[i + 1]]['weight']
	print("APPROX IS ", p)
	return sum


#make and show a complete graph on 10 vertices
def make_graph():
	G = nx.complete_graph(GRAPH_SIZE)
	generate_triangle_inequality_weight(G)
	#generate_random_weights(G)
	return G

#get version of graph G with w(ab) as -1 * w(ab)
def get_negative_weights(G):
	for e in G.edges():
		G[e[0]][e[1]]['weight'] = -1 * G[e[0]][e[1]]['weight']
	return G


#produce hamiltonian cycle for complete graph, using christofides algorithm
def christofides_algorithm(G):

	mst = find_mst_with_kruskal(G)
	show_graph(mst, 'minumum spanning tree')
	G_neg = get_negative_weights(G)
	even_degree = [v for v in mst.nodes() if mst.degree[v] % 2 == 0]
	odd_graph = G.copy()
	odd_graph.remove_nodes_from(even_degree)
	show_graph(odd_graph, 'induced graph by removing even vertices')
	min_weight_matching = nx.algorithms.max_weight_matching(odd_graph,weight='weight', maxcardinality=True)
	H = mst.copy()
	H = nx.MultiGraph()

	for e in mst.edges():
		H.add_edge(e[0],e[1],weight=mst[e[0]][e[1]]['weight'])

	for e in min_weight_matching:
		weight = G_neg[e[0]][e[1]]['weight'] * -1
		H.add_edge(e[0],e[1])
		H[e[0]][e[1]][len(H[e[0]][e[1]]) - 1]['weight'] = weight
	#show_graph(H)
	eulerian_circuit = nx.algorithms.eulerian_circuit(H)
	for c in eulerian_circuit:
		for e in c:
			print(e)
	show_multigraph(H)

#display a simple graph with title
def show_graph(G,title):
	plt.title(title)
	pos = nx.spring_layout(G)
	weights = nx.get_edge_attributes(G,'weight')
	nx.draw(G,pos)
	nx.draw_networkx_labels(G, pos)
	nx.draw_networkx_edge_labels(G,pos,weights, with_labels=True)
	plt.show()

#display a multigraph with the number of edges between two vertices max two as two subplots
def show_multigraph(G):
	A = nx.Graph()
	B = nx.Graph()
	last_e = False
	print(G.edges())
	for e in G.edges():
		print(G[e[0]][e[1]])
		if last_e == e:
			B.add_edge(last_e[0],last_e[1],weight=G[e[0]][e[1]][1]['weight'])
		else:
			A.add_edge(e[0],e[1],weight=G[e[0]][e[1]][0]['weight'])
		last_e = e
	
	plt.subplot(2,1,1)
	plt.title('MST')
	pos = nx.spring_layout(A)
	weights = nx.get_edge_attributes(A,'weight')
	nx.draw(A,pos)
	nx.draw_networkx_labels(A, pos)
	nx.draw_networkx_edge_labels(A,pos,weights, with_labels=True)
	
	plt.subplot(2,1,2)
	plt.title('Perfect matching from odd graph')
	pos = nx.spring_layout(B)
	weights = nx.get_edge_attributes(B,'weight')
	nx.draw(B,pos)
	nx.draw_networkx_labels(B, pos)
	nx.draw_networkx_edge_labels(B,pos,weights, with_labels=True)
	plt.show()


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

G = make_graph()
christofides_algorithm(G)
#mst = find_mst_with_kruskal(G)

#show_graph(mst)
#u = np.random.randint(GRAPH_SIZE)
#v = np.random.randint(GRAPH_SIZE)
#print("FINDING PATH FROM ",u, " TO ", v)
#approx_path(G,u,v)
#G = make_and_show_graph
#print("DJIKSTRA FOUND ",dijkstra_dist(G,u)[v])
#print("KRUSKAL FOUND ",approx_path(G,u,v))
#show_graph(G)
#show_graph(mst)
#plt.show(G)
#plt.show(mst)


exit()