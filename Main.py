import pandas as pd
import random as rd


######################################## BASIC FUNCTIONS ########################################################################################################################


def get_random_from_list(list1,list2):
    """"return a random elemnt from list1 that is not in list2"""
    element = rd.choice(list1)
    while element in list2:
        element = rd.choice(list1)
    return element


def get_random_from_list2(list1,list2):
    """"return a random elemnt from list1 that is not in list2 (used when the list is not too big)"""
    list_copy = list1[:]
    rd.shuffle(list_copy)
    for element in list_copy:
        if element not in list2:
            return element


######################################## DATA ########################################################################################################################


def get_matrix(path):
        """Read the file and return the data in a matrix"""
        matrix = []
        with open (path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                matrix.append(line[:len(line)-1])
        return matrix

def get_list(map_matrix, char):
    """Return a list of the coordinates of the habitations"""
    habitation_list = []
    for i in range(len(map_matrix)):
        for j in range(len(map_matrix)):
            if map_matrix[i][j] == char:
                habitation_list.append((i,j))
    return habitation_list


######################################## GEN ########################################################################################################################


def generate_parent(free_fields,budget,cost_matrix,max_iter):
    """Generate a parent"""
    parent = []
    parent_budget = 0
    while (parent_budget < budget):
        parent.append(get_random_from_list(free_fields,parent))
        cost = int(cost_matrix[parent[-1][0]][parent[-1][1]])
        if(parent_budget +cost > budget):
            parent.remove(parent[-1])
            for i in range(max_iter):
                field = get_random_from_list(free_fields,parent)
                cost = int(cost_matrix[field[0]][field[1]])
                if(parent_budget +cost <= budget):
                    parent.append(field)
                    parent_budget += cost
                    break
            break
        else:
            parent_budget += cost
    return parent


def generate_parents(free_fields,budget,cost_matrix,gen_length,max_iter):
    """Generate a list of parents"""
    parents = []
    for i in range(gen_length):
        parent = generate_parent(free_fields,budget,cost_matrix,max_iter)
        parents.append(parent)
    return parents


def generate_child(parent1,parent2,budget,cost_matrix,max_iter):
    """Generate a child from two parents"""
    child = []
    child_budget = 0
    parent1_budget = rd.choice(range(10,budget-10)) # Select a random budget that determines the number of fields passed by the 1st parent 
    #First part of the child
    while (child_budget < parent1_budget):
        child.append(get_random_from_list2(parent1,child))
        cost = int(cost_matrix[child[-1][0]][child[-1][1]])
        child_budget += cost
    #Second part of the child
    while (child_budget < budget):
        child.append(get_random_from_list2(parent2,child))
        cost = int(cost_matrix[child[-1][0]][child[-1][1]])
        if(child_budget + cost > budget):
            child.remove(child[-1])
            combined_parent = parent1 + parent2
            for i in range(max_iter):
                field = get_random_from_list2(combined_parent,child)
                cost = int(cost_matrix[field[0]][field[1]])
                if(child_budget + cost <= budget):
                    child.append(field)
                    child_budget += cost
            break
        else:
            child_budget += cost
    return child

def generate_child_second_methode(parents,free_fields,budget,cost_matrix):
    #ordonner les deux parents de sorte que les terrains sont localisés et couper a un endroit et mélanger les gènes ce qui donnera
    #des terrains déjà localisés pour les enfants 
    return None

def generate_new_gen(old_gen,score_list,elite_portion,mutate_rate,budget,cost_matrix,max_iter):
    """Generate a new generation from the old one"""
    new_gen = []
    production_score_list,compacity_score_list,average_distance_list = score_lists(score_list)
    #Elitism
    elite_length = int(len(old_gen)*elite_portion)
    for i in range(elite_length):
        random_nbr = rd.choice([1,2,3])
        if(random_nbr == 1):
            score_tuple = production_score_list.pop(0)
            new_gen.append(old_gen[score_tuple[0]])
        elif(random_nbr == 2):
            score_tuple = compacity_score_list.pop(0)
            new_gen.append(old_gen[score_tuple[0]])
        else:
            score_tuple = average_distance_list.pop(0)
            new_gen.append(old_gen[score_tuple[0]])
    #Crossover
    while len(new_gen) < len(old_gen):
        nums = rd.sample(range(0,elite_length),2)
        child = generate_child(old_gen[nums[0]],old_gen[nums[1]],budget,cost_matrix,max_iter)
        new_gen.append(child)
    #Mutation
    return new_gen



######################################## SCORE ########################################################################################################################


def average_distance(parent,habitation_list,distance_matrix):
    """Calculate the average distance between a field and the habitations"""
    total_distance = 0
    for field in parent:
        average_field_distance = 0 #average distance between one field and all the other habitations 
        if(distance_matrix[field[0]][field[1]] != -1):
            #if the distance between the field and the habitations has already been calculated
            average_field_distance = distance_matrix[field[0]][field[1]]
        else:
            total_field_distance = 0
            for j in range(len(habitation_list)):
                total_field_distance += abs(field[0] - habitation_list[j][0]) + abs(field[1] - habitation_list[j][1])
            average_field_distance = total_field_distance / len(habitation_list)
            distance_matrix[field[0]][field[1]] = average_field_distance   
        total_distance += average_field_distance
    average_distance = total_distance / len(parent)
    return average_distance


def production_score(parent,production_matrix):
    """Calculate the production score of a parent"""
    total_production = 0
    for i in range(len(parent)):
        total_production += int(production_matrix[parent[i][0]][parent[i][1]])
    return total_production


def compacity_score(parent):
    """Calculate the compacity score of a parent"""
    L = ( max(parent)[0] - min(parent)[0] ) + 1
    l = ( max(parent,key=lambda x:x[1])[1] - min(parent,key=lambda x:x[1])[1] ) + 1
    surface = L*l
    adjacent_score = 0
    for field in parent:
        if((field[0]-1,field[1]) in parent):
            adjacent_score += 1
        if((field[0]+1,field[1]) in parent):
            adjacent_score += 1
        if((field[0],field[1]-1) in parent):
            adjacent_score += 1
        if((field[0],field[1]+1) in parent):
            adjacent_score += 1
    return adjacent_score/surface


def calculate_score(parent,habitation_list,production_matrix,distance_matrix):
    """Calculate the score of a parent"""
    #Init
    prod_score = production_score(parent,production_matrix)
    habitation_average_distance = average_distance(parent,habitation_list,distance_matrix)
    comp_score = compacity_score(parent)
    score = (prod_score,habitation_average_distance,comp_score)
    return score


def calculate_gen_score(gen,habitation_list,production_matrix,distance_matrix):
    """Calculate the score of a generation"""
    score_list = []
    for parent in gen:
        score_list.append(calculate_score(parent,habitation_list,production_matrix,distance_matrix))
    return score_list


def score_lists(score_list):
    """Return three lists with the best production, habitation and compacity scores"""
    index_score_list = []
    for i in range(len(score_list)):
        index_score_list.append((i,score_list[i]))
    production_score_list = sorted(index_score_list,key=lambda x:x[1][0],reverse=True)
    habitation_score_list = sorted(index_score_list,key=lambda x:x[1][1])
    compacity_score_list = sorted(index_score_list,key=lambda x:x[1][2],reverse=True)
    return production_score_list,habitation_score_list,compacity_score_list



######################################## MAIN ########################################################################################################################


def main():
    #Parameters
    rd.seed(123)
    gen_length = 100
    gen_nbr = 10
    budget = 50
    elite_portion = 0.3
    mutate_rate = 0.5
    max_iter = 100
    #Read the data
    cost_matrix = get_matrix('donnes_V2\Cost_map.txt')
    production_matrix = get_matrix('donnes_V2\Production_map.txt')
    map_matrix = get_matrix('donnes_V2\\Usage_map.txt')
    habitation_list = get_list(map_matrix,'C')
    free_fields = get_list(map_matrix,' ')
    distance_matrix = [[-1 for i in range(len(map_matrix))] for j in range(len(map_matrix))] # a matrix to store the average distance between a field and the habitations
    #Init
    gen = generate_parents(free_fields,budget,cost_matrix,gen_length,max_iter)
    score_list = calculate_gen_score(gen,habitation_list,production_matrix,distance_matrix)
    for i in range(gen_nbr):
        new_gen = generate_new_gen(gen,score_list,elite_portion,mutate_rate,budget,cost_matrix,max_iter)
        new_score_list = calculate_gen_score(new_gen,habitation_list,production_matrix,distance_matrix)
        gen = new_gen[:]
        score_list = new_score_list[:]
        print(gen[0])
        print(score_list[0])
    

if __name__ == '__main__':
    main()