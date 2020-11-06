
def create_graph(food_needed):
   # create the graph for this

    def sort_min_distance(val):
        return abs(val[0][0] - val[1][0]) + abs(val[0][1] - val[1][1])
    graph = []
    seen_nodes = []
    final_food = []
    food_after = [[[i,x] for x in food_needed if i != x] for i in food_needed]
    for food in food_after:
        for val in food:
            final_food.append(val)
    final_food.sort(key=sort_min_distance)

    for food in final_food:
        if not (food[1], food[0]) in seen_nodes:
            seen_nodes.append((food[0], food[1]))
            graph.append([food[0], food[1], abs(food[0][0] - food[1][0]) + abs(food[0][1] - food[1][1])])

    return graph
def find_perm(graph, seen_list, current_pos, need_see, list_weight):
    # graph => u,v, weight ,
    if len(seen_list) == need_see:
        return seen_list, list_weight
    else:
        new_info = []
        for val in graph:
            if current_pos in val:
                new_seen = seen_list[:]
                new_weights = list_weight[:]
                u,v,weight = val
                if not(u in seen_list and v in seen_list):
                    new_weights.append(weight)
                    if u == current_pos:
                        new_seen.append(v)
                        new_pos = v
                    else:
                        new_seen.append(u)
                        new_pos = u
                    new_info.append([new_seen, new_weights, new_pos])
        overall_info = []
        for info in new_info:
            ordered_seen, listed_weights = find_perm(graph, info[0], info[2], need_see, info[1])
            overall_info.append([ordered_seen, listed_weights])
        min_1 = 9328409099999999999999999999
        final_order = []
        final_min = []
        for new_info in overall_info:
            if sum(new_info[1]) < min_1:
                min_1 = sum(new_info[1])
                final_min = new_info[1]
                final_order = new_info[0]
        return final_order, final_min
if "__main__" == __name__:
    current = (2,3)
    graph = create_graph([(0,0),(2,0), (6,11), (4,1)])
    for c in [(0,0),(2,0), (6,11), (4,1)]:
        order, min_order = find_perm(graph, [c], c, 4, [0])
        print(order, min_order)
        print(sum(min_order) + (abs(current[0] - c[0]) + abs(current[1] - c[1])))


