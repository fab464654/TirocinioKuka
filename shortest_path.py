# Source: https://www.geeksforgeeks.org/traveling-salesman-problem-tsp-implementation/
from sys import maxsize
from itertools import permutations
from scipy.spatial.distance import cdist
import cv2

# implementation of traveling Salesman Problem
def travellingSalesmanProblem(graph, source_vertex):
    V = len(graph)

    # store all vertex apart from source vertex
    vertex = []
    for i in range(V):
        if i != source_vertex:
            vertex.append(i)

    # store minimum weight Hamiltonian Cycle
    min_path = maxsize
    all_permutations = permutations(vertex)
    best_permutation = ()  #permutation to return that has the lowest cost
    best_costs = []
    for permutation_to_test in all_permutations:
        current_costs = []
        # store current Path weight (cost)
        current_pathweight = 0

        # compute current path weight
        k = source_vertex
        for curr_node in permutation_to_test:
            #print("permutation_to_test=",permutation_to_test, "curr_node=",curr_node, "current_pathweight=", current_pathweight)
            current_pathweight += graph[k][curr_node]
            current_costs.append(graph[k][curr_node])  #list that contains the costs for the current permutation
            k = curr_node

        #current_pathweight += graph[curr_node][source_vertex]  #uncomment if it's needed to go back to the source vertex
        #print("current_pathweight=",current_pathweight)

        # update minimum
        prev_min_path = min_path
        min_path = min(min_path, current_pathweight)

        if min_path < prev_min_path:
            best_permutation = permutation_to_test  #best permutation found (min. cost)
            best_costs = current_costs  #costs associated to the best permutation

    best_permutation = list(best_permutation)
    best_permutation.insert(0, source_vertex)

    return best_permutation, best_costs, min_path

def create_graph_from_coords(centers):
    # matrix representation of graph
    graph = []
    for c1 in centers:
        distances_from_c1 = []
        for c2 in centers:
            dist = cdist([c1[0:2]], [c2[0:2]])
            dist = dist.tolist()
            dist = dist[0][0]
            #print("dist:", c1[0:2], c2[0:2], dist)
            distances_from_c1.append(dist)
        graph.append(distances_from_c1)

    #print(graph)
    # Graph structure example:
    #graph = [[0, 10, 15, 20], [10, 0, 35, 25],
    #         [15, 35, 0, 30], [20, 25, 30, 0]]
    return graph

def plot_graph_over_image(img, graph, centers, best_permutation, best_costs):

    #best_costs_cm = [dist_pixel/96*2.54 for dist_pixel in best_costs]
    best_costs_cm = [dist_pixel / 60 * 2.54 for dist_pixel in best_costs]
    print("All costs in [cm]:", best_costs_cm)

    pos_text_x = 0

    font = cv2.FONT_HERSHEY_SIMPLEX

    for cost_cm in best_costs_cm:
        pos_text_x += 80
        org = (pos_text_x, 50)
        fontScale = 0.5
        color = (0, 0, 255)
        thickness = 2
        cv2.putText(img, " -> " + str(round(cost_cm,2)) + "cm ", org, font, fontScale, color, thickness, cv2.LINE_AA)

    path_index = 0
    for current_node in best_permutation:
        path_index += 1
        cv2.circle(img, (int(centers[current_node][0]), int(centers[current_node][1])), int(centers[current_node][2]), (0, 0, 255), -1)
        cv2.line(img, (int(centers[current_node][0]) - 20, int(centers[current_node][1])), (int(centers[current_node][0]) + 20, int(centers[current_node][1])), (0, 0, 0), 3)
        cv2.line(img, (int(centers[current_node][0]), int(centers[current_node][1]) - 20), (int(centers[current_node][0]), int(centers[current_node][1]) + 20), (0, 0, 0), 3)

        org = (int(centers[current_node][0] * 1.05), int(centers[current_node][1]*0.95))
        fontScale = 0.8
        color = (0, 128, 255)
        thickness = 2
        cv2.putText(img, str(path_index), org, font, fontScale, color, thickness, cv2.LINE_AA)

        org = (int(centers[current_node][0] * 1.05), int(centers[current_node][1]*1.05))
        fontScale = 0.4
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(img, str(int(centers[current_node][0])) + "; " + str(int(centers[current_node][1])), org, font, fontScale, color, thickness, cv2.LINE_AA)


    cv2.imshow("Resulting graph order", img)
