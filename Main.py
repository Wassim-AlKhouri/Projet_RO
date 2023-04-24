import pandas as pd
import random as rd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


######################################## BASIC FUNCTIONS ########################################################################################################################


def get_random_from_list(list1,list2):
    """"return a random elemnt from list1 that is not in list2"""
    element = rd.choice(list1)
    while element in list2:
        element = rd.choice(list1)
    return element


def budget_check(child,free_fields,cost_matrix,child_budget,budget):
    """Check if the budget is respected after a mutation"""
    #TODO : corriger la fonction car budget trop bas par fois
    if (child_budget > budget):
        child_budget -= int(cost_matrix[child[-1][0]][child[-1][1]])
        child.remove(child[-1])
        child.append(get_random_from_list(free_fields,child))
        child_budget += int(cost_matrix[child[-1][0]][child[-1][1]])
        child_budget = budget_check(child,free_fields,cost_matrix,child_budget,budget)   
    return child_budget


def get_random_from_list2(list1,list2):
    """"
    Returns a bool indicating if a field from list1 that is not in list2 was found and the field if it was found.
    This method is used with small a list1.
    """
    found = False
    found_field = None
    list1_copy = list1[:]
    rd.shuffle(list1_copy)
    for field in list1_copy:
        if field not in list2:
            found = True
            found_field = field
            break
    return (found,found_field)


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
            #If the budget is exceeded, remove the last field and try again 
            #with a maximum of max_iter tries to find a new one
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


def generate_child(parent1,parent2,budget,cost_matrix,max_iter,mutate_rate,free_fields):
    """Generate a child from two parents"""
    child = []
    child_budget = 0
    parent1_budget = rd.choice(range(10,budget-10)) # Select a random budget that determines the number of fields passed by the 1st parent 
    #First part of the child
    while (child_budget < parent1_budget and len(child) < len(parent1)):
        found,field =get_random_from_list2(parent1,child) 
        if(not found) : break
        child.append(field)
        child_budget += int(cost_matrix[child[-1][0]][child[-1][1]])
    #Second part of the child
    while (child_budget < budget and len(child) < len(parent1) + len(parent2)):
        found,field = get_random_from_list2(parent2,child)
        if(not found) : break
        child.append(field)
        cost = int(cost_matrix[child[-1][0]][child[-1][1]])
        if(child_budget + cost > budget):
            child.remove(child[-1])
            combined_parent = parent1 + parent2
            for i in range(max_iter):
                found,field = get_random_from_list2(combined_parent,child)
                if(not found) : break
                cost = int(cost_matrix[field[0]][field[1]])
                if(child_budget + cost <= budget):
                    child.append(field)
                    child_budget += cost
            break
        else:
            child_budget += cost
    #Mutation
        if(rd.random() < mutate_rate):
            field_to_mutate = rd.choice(child)
            child_budget -= int(cost_matrix[field_to_mutate[0]][field_to_mutate[1]])
            child.remove(field_to_mutate)
            child.append(get_random_from_list(free_fields,child))
            child_budget += int(cost_matrix[child[-1][0]][child[-1][1]])
            child_budget = budget_check(child,free_fields,cost_matrix,child_budget,budget)
    return child


def generate_new_gen(old_gen,score_list,elite_portion,mutate_rate,budget,cost_matrix,max_iter,free_fields):
    """Generate a new generation from the old one"""
    new_gen = []
    production_score_list,average_distance_list,compacity_score_list = score_lists(score_list)
    #Elitism
    elite_length = int(len(old_gen)*elite_portion)
    for i in range(elite_length):
        random_nbr = rd.choice([1,2,3])
        if(random_nbr == 1):
            score_tuple = production_score_list.pop(0)
            new_gen.append(old_gen[score_tuple[0]])
        elif(random_nbr == 2):
            score_tuple = average_distance_list.pop(0)
            new_gen.append(old_gen[score_tuple[0]])
        else:
            score_tuple = compacity_score_list.pop(0)
            new_gen.append(old_gen[score_tuple[0]])
    #Crossover
    while len(new_gen) < len(old_gen):
        nums = rd.sample(range(0,elite_length),2)
        child = generate_child(old_gen[nums[0]],old_gen[nums[1]],budget,cost_matrix,max_iter,mutate_rate,free_fields)
        new_gen.append(child)
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
    adjacent_score = 1
    for field in parent:
        if((field[0]-1,field[1]) in parent):
            adjacent_score += 1
        if((field[0]+1,field[1]) in parent):
            adjacent_score += 1
        if((field[0],field[1]-1) in parent):
            adjacent_score += 1
        if((field[0],field[1]+1) in parent):
            adjacent_score += 1
    return surface/adjacent_score


def calculate_score(parent,habitation_list,production_matrix,distance_matrix):
    """Calculate the score of a parent, returns a tuple (production,habitation,compacity)"""
    #Init
    prod_score = production_score(parent,production_matrix)
    habitation_average_distance = average_distance(parent,habitation_list,distance_matrix)
    comp_score = compacity_score(parent)
    score = (prod_score,habitation_average_distance,comp_score)
    return score


def calculate_gen_score(gen,habitation_list,production_matrix,distance_matrix):
    """Calculate the score of a generation, returns a list of tuples (production,habitation,compacity)"""
    score_list = []
    for parent in gen:
        score_list.append(calculate_score(parent,habitation_list,production_matrix,distance_matrix))
    return score_list


def score_lists(score_list):
    """
    Return three lists with the best production, habitation and compacity scores.
    Each list is a list of tuples (index,score)
    """
    index_score_list = []
    for i in range(len(score_list)):
        index_score_list.append((i,score_list[i]))
    production_score_list = sorted(index_score_list,key=lambda x:x[1][0],reverse=True)
    habitation_score_list = sorted(index_score_list,key=lambda x:x[1][1])
    compacity_score_list = sorted(index_score_list,key=lambda x:x[1][2])
    return production_score_list,habitation_score_list,compacity_score_list


######################################## PARETO ########################################################################################################################


def init_pareto(score_list,gen):
    """Init the pareto list, returns a list of tuples (parent,(production,habitation,compacity))"""
    pareto_list = []
    for i in range(len(score_list)):
        is_pareto = True
        for solution in pareto_list:
            if(solution[1][0] >= score_list[i][0] and solution[1][1] <= score_list[i][1] and solution[1][2] <= score_list[i][2]):
                is_pareto = False
                break
        if(is_pareto):
            pareto_list.append((gen[i],score_list[i]))
    return pareto_list


def update_pareto(pareto_list,score_list,gen):
    """Update the pareto list, returns a list of tuples (parent,(production,habitation,compacity))"""
    bad_solutions = []
    for i in range(len(score_list)):
        is_pareto = True
        for solution in pareto_list:
            if (solution[1][0] >= score_list[i][0] and solution[1][1] <= score_list[i][1] and solution[1][2] <= score_list[i][2]):
                is_pareto = False
                break
            if (solution[1][0] <= score_list[i][0] and solution[1][1] >= score_list[i][1] and solution[1][2] >= score_list[i][2]):
                if(solution not in bad_solutions):
                    bad_solutions.append(solution)
        if(is_pareto):
            pareto_list.append((gen[i],score_list[i]))
    #Remove bad solutions
    for bad_solution in bad_solutions:
        pareto_list.remove(bad_solution)
    return pareto_list


def test_pareto(pareto_list):
    """Test if the pareto list is correct"""
    for i in range(len(pareto_list)):
        for j in range(len(pareto_list)):
            if(i != j):
                if(pareto_list[i][1][0] >= pareto_list[j][1][0] and pareto_list[i][1][1] <= pareto_list[j][1][1] and pareto_list[i][1][2] <= pareto_list[j][1][2]):
                    print("Pareto error")
                    return False
    return True


######################################## PROMETHEE ########################################################################################################################

def promethee(score1,score2,weights,preference):
    """Calculate the promethee score of a solution"""
    score = 0
    for i in range(len(score1)):
        difference = score1[i] - score2[i]
        if(difference > 0):
            if(difference > preference[i][0]):
                if(difference < preference[i][1]):
                    score += weights[i]*(difference-preference[i][0])/(preference[i][1]-preference[i][0])
                else:
                    score += weights[i]
    return score


def create_promethee_list(pareto_list,score_list,weights,preference):
    """Create the promethee list"""
    promethee_list =[[0 for i in range(len(pareto_list))] for j in range(len(pareto_list))]
    for i in range(len(pareto_list)):
        promethee_list[i][i] = 0
        for j in range(i+1,len(pareto_list)):
            promethee_list[i][j] = promethee(score_list[i],score_list[j],weights,preference)
            promethee_list[j][i] = promethee(score_list[j],score_list[i],weights,preference)
    return promethee_list


def calculate_promethee_score(promethee_list):
    """Calculate the promethee score of a generation, returns a list of tuples (index,score)"""
    promethee_score_list = []
    for i in range(len(promethee_list)):
        positive_score = sum(promethee_list[i])
        negative_score = 0
        for j in range(len(promethee_list)):
            negative_score += promethee_list[j][i]
        promethee_score_list.append((i,positive_score-negative_score))
    return promethee_score_list

######################################## MAIN ########################################################################################################################


def main():
    #Parameters
    #rd.seed()
    gen_length = 500  #number of parents in a generation
    gen_nbr = 500  #number of generations
    budget = 50
    elite_portion = 0.3  #portion of the best parents that will be kept in the next generation
    mutate_rate = 0.05  #rate of mutation
    max_iter = 100  #maximum number of tries to find a field that fits the budget
    #Promethee parameters
    weights = (0.5,0.5,0.5) #weights of the criteria (production,habitation,compacity)
    preference = ((0.5,1),(0.5,1),(0.5,1)) #preference of the criteria (min,max) (production,habitation,compacity)
    #Read the data
    cost_matrix = get_matrix('donnes_V2\Cost_map.txt')
    production_matrix = get_matrix('donnes_V2\Production_map.txt')
    map_matrix = get_matrix('donnes_V2\\Usage_map.txt')
    habitation_list = get_list(map_matrix,'C')
    free_fields = get_list(map_matrix,' ')
    distance_matrix = [[-1 for i in range(len(map_matrix))] for j in range(len(map_matrix))] # a matrix to store the average distance between a field and the habitations
    #Init gen and score_list
    gen = generate_parents(free_fields,budget,cost_matrix,gen_length,max_iter)
    score_list = calculate_gen_score(gen,habitation_list,production_matrix,distance_matrix)
    #Init plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #Init pareto
    pareto_list = init_pareto(score_list,gen) #list of pareto solutions = (parent,score)
    #Main Loop
    for i in range(gen_nbr):
        #Genrate the new generation and calculate the score of the new generation
        new_gen = generate_new_gen(gen,score_list,elite_portion,mutate_rate,budget,cost_matrix,max_iter,free_fields)
        new_score_list = calculate_gen_score(new_gen,habitation_list,production_matrix,distance_matrix)
        #Plot
        pareto_list = update_pareto(pareto_list,new_score_list,new_gen)
        gen = new_gen[:]
        score_list = new_score_list[:]
        print("Generation",i+1,"/",gen_nbr)
    solutions,coordinates = zip(*pareto_list)
    #Promethee
    promethee_list = creat_promethee_list(pareto_list,weights,preference)

    x,y,z = zip(*coordinates)
    ax.scatter(x,y,z)
    ax.set_xlabel('Production')
    ax.set_ylabel('Habitation')
    ax.set_zlabel('Compacity')
    prod,hab,comp = score_lists(coordinates)
    print("Best production score:",prod[0][1][0])
    print("Best habitation score:",hab[0][1][1])
    print("Best compacity score:",comp[0][1][2])
    print(solutions[comp[0][0]])
    print(test_pareto(pareto_list))
    plt.show()


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("Time:",end-start)