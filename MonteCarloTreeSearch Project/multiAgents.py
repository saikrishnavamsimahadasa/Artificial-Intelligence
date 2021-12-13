# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance, Counter
from game import Directions
import random, util
import math
import time
from game import Agent
from ghostAgents import DirectionalGhost
from ghostAgents import RandomGhost
from featureExtractors import SimpleExtractor                                            

class Node():
    """stores nodes in the search tree."""
    id_node = 0
    
    def __init__(self, state, action, parent, agent_index=0):

        self.state = state
        self.agent_index = agent_index
        self.parent = parent
        self.action = action
        self.wins_num = 0
        self.explored_num = 0 
        self.sum_score = 0 
        self.id_node = Node.id_node 
        Node.id_node += 1
        self.children = [] 

    def win_best(self):
        scores = [1.0 * child.wins_num / child.explored_num if child.explored_num else 0.0 for child in self.children]
        max_score = max(scores)
        max_index = [i for i in range(len(scores)) if scores[i] == max_score]
        index_chosen = random.choice(max_index) 
        return self.children[index_chosen]
    
    def visited_most(self):
        scores = [child.explored_num for child in self.children]
        max_score = max(scores)
        max_index = [i for i in range(len(scores)) if scores[i] == max_score]
        index_chosen = random.choice(max_index) 
        return self.children[index_chosen]
    
    def score_best(self):
        scores = [1.0 * child.sum_score / child.explored_num if child.explored_num else 0.0 for child in self.children]
        max_score = max(scores)
        max_index = [i for i in range(len(scores)) if scores[i] == max_score]
        index_chosen = random.choice(max_index) 
        return self.children[index_chosen] 
       
    def win_score_best(self):
        max_index = []
        best_score = -float('inf')
        for c_child in self.children:
            if c_child.explored_num:
                winaverage = c_child.wins_num / c_child.explored_num
                if winaverage > 0.1: c_score = (winaverage * c_child.sum_score) / c_child.explored_num
                else: c_score = winaverage
            else: c_score = -float('inf')
            if c_score > best_score:
                max_index = [c_child]
                best_score = c_score
            elif c_score == best_score:
                max_index.append(c_child)
        return random.choice(max_index)
    
    def find_action(self, b_child_algorithm='best_combination'):
        if b_child_algorithm == 'best_win': b_child = self.win_best()
        elif b_child_algorithm == 'best_combination': b_child = self.win_score_best()
        elif b_child_algorithm == 'most_visited': b_child = self.visited_most()
        else: b_child = self.score_best()
        return b_child.action
    
    def upperconfidencebound(self, c=150.0):
        scores = [(1.0 * child.sum_score / child.explored_num) + (c * (math.log(self.explored_num)/child.explored_num)) if child.explored_num else float('inf') for child in self.children]
        max_score = max(scores)
        max_index = [i for i in range(len(scores)) if scores[i] == max_score]
        index_chosen = random.choice(max_index) 
        return self.children[index_chosen]
        
    def epsilongreed(self, exploit_weight=0.8):
        if random.random() < exploit_weight:
            scores = [1.0 * child.sum_score / child.explored_num if child.explored_num else 0.0 for child in self.children]
            max_score = max(scores)
            max_index = [i for i in range(len(scores)) if scores[i] == max_score]
            index_chosen = random.choice(max_index) 
        else: index_chosen = random.choice(range(len(self.children)))
        return self.children[index_chosen]
    
    def exploit_explore(self, e_algorithm='ucb', e_variable=''):
        if e_algorithm == 'ucb':
            if e_variable == '': return self.upperconfidencebound()
            else: return self.upperconfidencebound(float(e_variable))
        else:
            if e_variable == '': return self.epsilongreed()
            else: return self.epsilongreed(float(e_variable))
    
    def generate_children(self):
        legal_moves = self.state.getLegalActions(self.agent_index)
        children = []
        for i in range(len(legal_moves)):
            action = legal_moves[i]
            child_state = self.state.generateSuccessor(self.agent_index, action)
            new_child = Node(child_state, action, parent=self, agent_index=(self.agent_index+1) % self.state.getNumAgents() )
            children.append(new_child)
        self.children = children
    
    def score_update(self, win, score):
        self.explored_num +=1
        if self.agent_index == 1:
            self.wins_num +=  float(win)
            self.sum_score += score
        else:
            self.wins_num -= float(not win)
            self.sum_score -= score
            
    def print_structure(self, t=0):
        print(" " * t + "ID", self.id_node)
        if self.parent: print(" " * t + "Parent", self.parent.id_node)
        else: print(" " * t + "ROOT")
        print(" " * t + "Agent index", self.agent_index)
        print(" " * t + "Wins", self.wins_num)
        print(" " * t + "Score", self.sum_score)
        print(" " * t + "Explored", self.explored_num)
        for child in self.children:
            child.print_structure(t+2)


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newFoodList = newFood.asList()
        food_dist = math.inf
        newGhostPositions = successorGameState.getGhostPositions()
        
        for ghost in newGhostPositions:
            ghost_dist = manhattanDistance(newPos, ghost)
            if (ghost_dist < 2):
                return -1
        
        for food in newFoodList:
            actual_dist = manhattanDistance(newPos, food)
            food_dist = min(food_dist,actual_dist)
        
        # Evaluation function
        food_dist_reverse = 1.0/food_dist
        func_val = successorGameState.getScore() + food_dist_reverse
        return func_val

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    number_of_nodes = []
    depth_of_tree = []
    time_per_moves = []                                          
    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '4'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    current_number_of_nodes = 0
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #eval function value at root node - pacman
        start_time = time.time()
        retur_val = self.maxim(gameState, 0, 0)
        end_time = time.time()
        MultiAgentSearchAgent.time_per_moves.append(end_time - start_time)
        MultiAgentSearchAgent.number_of_nodes.append(self.current_number_of_nodes)
        MultiAgentSearchAgent.depth_of_tree.append(self.depth * gameState.getNumAgents())
        self.current_number_of_nodes = 0
        return retur_val[0]     #returns action

    def maxim(self, gameState, depth, agent_index):
        self.current_number_of_nodes += 1
        agent_num = gameState.getNumAgents()
        compare_val = ("", -math.inf)
        legal_actions = gameState.getLegalActions(agent_index)
        for act in legal_actions:
            i = depth + 1
            successor_state = gameState.generateSuccessor(agent_index, act)
            agent_ind = i % agent_num   #for pacman, index will be always Zero 
            minimax_val = self.processminimax(successor_state, i, agent_ind)
            
            #checks for the maximum value
            if minimax_val > compare_val[1]:
                maxim_val = (act, minimax_val)
                compare_val = maxim_val
            else:
                maxim_val = compare_val
                
        return maxim_val

    def minim(self, gameState, depth, agent_index):
        self.current_number_of_nodes += 1
        agent_num = gameState.getNumAgents()
        compare_val = ("", math.inf)
        legal_actions = gameState.getLegalActions(agent_index)
        for act in legal_actions:
            i = depth + 1
            successor_state = gameState.generateSuccessor(agent_index, act)
            agent_ind = i % agent_num       #for pacman, index will be always Zero 
            minimax_val = self.processminimax(successor_state, i, agent_ind)
            
            #checks for the minimum value
            if minimax_val < compare_val[1]:
                minim_val = (act, minimax_val)
                compare_val = minim_val
            else:
                minim_val = compare_val
        
        return minim_val
    
    def processminimax(self, gameState, depth, agent_index):
        agent_num = gameState.getNumAgents()
        #depth == 4 indicates four times of all the agents
        if gameState.isLose() or gameState.isWin() or depth == (agent_num * self.depth):
            return self.evaluationFunction(gameState)
        
        if agent_index == 0:
            #pacman eval function values
            maxim = self.maxim(gameState, depth, agent_index)
            return maxim[1]
        else:
            #ghost eval function values
            minim = self.minim(gameState, depth, agent_index)
            return minim[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    current_number_of_nodes = 0                                                                                                    
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #eval function value at root node - pacman
        start_time = time.time()
        retur_val = self.maxim(gameState, 0, 0, -math.inf, math.inf)
        end_time = time.time()
        MultiAgentSearchAgent.time_per_moves.append(end_time - start_time)
        MultiAgentSearchAgent.number_of_nodes.append(self.current_number_of_nodes)
        MultiAgentSearchAgent.depth_of_tree.append(self.depth * gameState.getNumAgents())
        self.current_number_of_nodes = 0
        return retur_val[0]     #returns action

    def maxim(self, gameState, depth, agent_index, a, b):
        self.current_number_of_nodes += 1
        agent_num = gameState.getNumAgents()
        compare_val = ("", -math.inf)
        legal_actions = gameState.getLegalActions(agent_index)
        for act in legal_actions:
            i = depth + 1
            successor_state = gameState.generateSuccessor(agent_index, act)
            agent_ind = i % agent_num   #for pacman, index will be always Zero 
            processalphabeta_val = self.processalphabeta(successor_state, i, agent_ind, a, b)
            
            #checks for the maximum value
            if processalphabeta_val > compare_val[1]:
                maxim_val = (act, processalphabeta_val)
                compare_val = maxim_val
            else:
                maxim_val = compare_val
                
            if maxim_val[1] > b: 
                return maxim_val
            else: 
                a = max(a, maxim_val[1])
                
        return maxim_val

    def minim(self, gameState, depth, agent_index, a, b):
        self.current_number_of_nodes += 1
        agent_num = gameState.getNumAgents()
        compare_val = ("", math.inf)
        legal_actions = gameState.getLegalActions(agent_index)
        for act in legal_actions:
            i = depth + 1
            successor_state = gameState.generateSuccessor(agent_index, act)
            agent_ind = i % agent_num   #for pacman, index will be always Zero 
            processalphabeta_val = self.processalphabeta(successor_state, i, agent_ind, a, b)
            
            #checks for the minimum value
            if processalphabeta_val < compare_val[1]:
                minim_val = (act, processalphabeta_val)
                compare_val = minim_val
            else:
                minim_val = compare_val
                
            if minim_val[1] < a: 
                return minim_val
            else: 
                b = min(b, minim_val[1])
        
        return minim_val
    
    def processalphabeta(self, gameState, depth, agent_index, a, b):
        agent_num = gameState.getNumAgents()
        #depth == 4 indicates four times of all the agents
        if gameState.isLose() or gameState.isWin() or depth == (agent_num * self.depth):
            return self.evaluationFunction(gameState)
        
        if agent_index == 0:
            #pacman eval function values
            maxim = self.maxim(gameState, depth, agent_index, a, b)
            return maxim[1]
        else:
            #ghost eval function values
            minim = self.minim(gameState, depth, agent_index, a, b)
            return minim[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    current_number_of_nodes = 0                                         
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        agent_num = gameState.getNumAgents()
        total_depth = agent_num * self.depth
        start_time = time.time()
        retur_val = self.maxim(gameState, total_depth, 0, "changenodeaction")
        end_time = time.time()
        MultiAgentSearchAgent.time_per_moves.append(end_time - start_time)
        MultiAgentSearchAgent.number_of_nodes.append(self.current_number_of_nodes)
        MultiAgentSearchAgent.depth_of_tree.append(total_depth)
        self.current_number_of_nodes = 0
        return retur_val[0]     #returns action

    def maxim(self, gameState, depth, agent_index, act):
        self.current_number_of_nodes += 1
        agent_num = gameState.getNumAgents()
        compare_val = ("", -math.inf)
        legal_actions = gameState.getLegalActions(agent_index)
        for acts in legal_actions:
            action_successor = None
            agent_ind = (agent_index + 1) % agent_num   #for pacman, index will be always Zero
            i = depth - 1
            successor_state = gameState.generateSuccessor(agent_index, acts)
            if depth != agent_num * self.depth:
                action_successor = act
            else:
                action_successor = acts
            expectimax_val = self.processexpectimax(successor_state, i, agent_ind, action_successor)
            
            #checks for the maximum value
            if expectimax_val > compare_val[1]:
                maxim_val = (acts, expectimax_val)
                compare_val = maxim_val
            else:
                maxim_val = compare_val
                
        return maxim_val

    def expect(self, gameState, depth, agent_index, act):
        self.current_number_of_nodes += 1    
        agent_num = gameState.getNumAgents()
        legal_actions = gameState.getLegalActions(agent_index)
        propability_val = 1.0/len(legal_actions)
        expect_val = 0
        for acts in legal_actions:
            i = depth - 1
            successor_state = gameState.generateSuccessor(agent_index, acts)
            agent_ind = (agent_index + 1) % agent_num      #for pacman, index will be always Zero 
            expectimax_val = self.processexpectimax(successor_state, i, agent_ind, act)
            
            expect_val += expectimax_val * propability_val

        return (act,expect_val)
    
    def processexpectimax(self, gameState, depth, agent_index, act):
        #depth == 4 indicates four times of all the agents
        if gameState.isLose() or gameState.isWin() or depth == 0:
            return self.evaluationFunction(gameState)
        
        if agent_index == 0:
            #pacman eval function values
            maxim = self.maxim(gameState, depth, agent_index, act)
            return maxim[1]
        else:
            #ghost eval function values
            expect = self.expect(gameState, depth, agent_index, act)
            return expect[1]
            
def better_Eval_Function(currentGameState):
    food = currentGameState.getFood()
    position = currentGameState.getPacmanPosition()
    ghost_states = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    scared_num = [g.scaredTimer for g in ghost_states]
    ghost_pos = [g.getPosition() for g in ghost_states]
    val_score = -1*food.count()
    val_score += -1*len(capsules)
    dist_ghosts = [util.manhattanDistance(position,g_pos) for g_pos in ghost_pos]
    
    for i in range(len(dist_ghosts)):
        if scared_num[i] > dist_ghosts[i]: val_score += 1
        else:
            if dist_ghosts[i] < 2: val_score += -100
                            
    if food.count() > 0:
        pos_food = food.asList()
        min_dist_food = min([util.manhattanDistance(position,f_pos) for f_pos in pos_food])
        val_score += -1*min_dist_food
    return val_score + currentGameState.getScore()

better = better_Eval_Function

class MonteCarloTreeSearchAgent(MultiAgentSearchAgent):
    current_number_of_nodes = 0
    c_tree = None

    def __init__(self, steps='500', reuse='True', simDepth='10', chooseAlg='best_combination', exploreAlg='ucb', exploreVar='', randSim='False', 
                    pacmanEps='0.9', earlyStop='True', tillBored='80', optimism='0.2', panics='True', simRelevance='0.1', dangerZone='0.2'):
                 
        self.allowed_steps = int(steps)  
        self.depth_simulation = int(simDepth)
        self.structure_reuse = reuse == 'True'  
        self.pacman_epsilon = float(pacmanEps)  
        self.near_stop = earlyStop == 'True'
        self.tillbored_steps = int(tillBored)
        self.explore_act = exploreAlg  
        self.e_algo_var = exploreVar  
        self.move_random = randSim == 'True'  
        self.bool_panics = panics == 'True'  
        self.sim_relevance = float(simRelevance)
        self.panic_rate = float(dangerZone) 
        self.feat_extractor = SimpleExtractor()
        self.choose_algo = chooseAlg
        self.epsilon_ghost = float(optimism)
        self.weights = Counter({'closest-food': -22.920237767606736, '#-of-ghosts-1-step-away': -2442.2537145683605, 'eats-food': 326.615053847113, 'bias': 0.6124765039597753}) 

    def getAction(self, gameState):
        
        def transit_rand(state, agent_index):
            legal_moves = state.getLegalActions(agent_index)
            if legal_moves:
                select_act = random.choice(legal_moves) 
                return state.generateSuccessor(agent_index, select_act), select_act
            else: return None
            
        def q_learning(state):
            legal_moves = state.getLegalActions(0)
            if legal_moves:
                moves_max = [legal_moves[0]]
                score_max = self.weights*self.feat_extractor.getFeatures(state,legal_moves[0])
                for move_current in legal_moves[1:]:
                    score_current = self.weights*self.feat_extractor.getFeatures(state,move_current)
                    if score_max < score_current:
                        moves_max = [move_current]
                        score_max = score_current
                    elif score_max == score_current: moves_max.append(move_current)
                select_act = random.choice(moves_max)
                return state.generateSuccessor(0, select_act), select_act
            else: return None
            
        def epsilon_greedy(state, e=0.9, agent_index=0):
            legal_moves = state.getLegalActions(agent_index)
            if legal_moves:
                if random.random() < e:
                    scores = [state.generateSuccessor(agent_index, a).getScore() for a in legal_moves]
                    max_index = [i for i in range(len(scores)) if scores[i] == max(scores)]
                    index_chosen = random.choice(max_index) 
                    select_act = legal_moves[index_chosen]
                    return state.generateSuccessor(agent_index, select_act), select_act
                else:
                    select_act = random.choice(legal_moves) 
                    return state.generateSuccessor(agent_index, select_act), select_act
            else: return None

        def leaf_select(tree):
            if not tree.children: return tree
            b_child = tree.exploit_explore(self.explore_act,self.e_algo_var)
            return leaf_select(b_child)

        def leaf_expand(leaf):
            leaf.generate_children()
            self.current_number_of_nodes += len(leaf.children)
                
        def backpropagation(result, node):
            win, score = result
            #update the score
            node.score_update(win, score)
            if node.parent is None: return
            backpropagation(result, node.parent)

        def heuristic_func(state):
            closest_food = float('inf')
            food = state.getFood()
            position = state.getPacmanPosition()
            for c_food in food.asList():
                distance = manhattanDistance(position, c_food)
                if closest_food > distance: closest_food = distance
            return 0.5, (state.getScore() * 0.5) + 400 - (closest_food * 0.25)
            
        def heuristic_func1(state):
            capsules = state.getCapsules()
            position = state.getPacmanPosition()
            food = state.getFood()
            ghost_states = state.getGhostStates()
            scared_times = [g.scaredTimer for g in ghost_states]
            ghost_positions = [g.getPosition() for g in ghost_states]
            score_val = -1*food.count()
            score_val += -1*len(capsules)
            dist_ghost = [util.manhattanDistance(position,g) for g in ghost_positions]
            for i in range(len(dist_ghost)):
                if scared_times[i] > dist_ghost[i]: score_val += 1
                else:
                    if dist_ghost[i] < 2: score_val += -100
            if food.count() > 0:
                food_pos = food.asList()
                min_food_dist = min([util.manhattanDistance(position,f) for f in food_pos])
                score_val += -1*min_food_dist
            return 0.5, score_val + state.getScore()
        
        def learned_heuristic(state):
            weights = Counter({'closest-food': -2.9590833461811363, 'bias': 205.60863391209026, '#-of-ghosts-1-step-away': -119.89950003939676, 'eats-food': 270.2008225113668})
            legalMoves = state.getLegalActions(0)
            if legalMoves:
                score = max([weights*self.featExtractor.getFeatures(state,a) for a in legalMoves])
                return 0.5, score
            else:
                return state.isWin(), state.getScore()
        
        def find_depth_max(node, c_depth=0):
            depth_max = c_depth
            if len(node.children) > 0:
                depth_max = find_depth_max(node.children[0], c_depth + 1)
                for c_child in range(1, len(node.children)):
                    depth_of_child = find_depth_max(node.children[c_child], c_depth + 1)
                    if depth_of_child > depth_max: depth_max = depth_of_child
            return depth_max

        def simulate_rand_act(node, agent_index=0, random_moves=False, heuristic_fn=heuristic_func):
            if random_moves:
                agent_index = 1
                state = node.state
                for c_turn in range(self.depth_simulation):
                    while agent_index < state.getNumAgents():
                        if state.isWin() or state.isLose(): return state.isWin(), state.getScore()
                        state, _ = transit_rand(state, agent_index)
                        agent_index += 1
                    agent_index = 0
                return heuristic_fn(state)
            else:
                state = node.state
                if random.random() < self.epsilon_ghost: ghosts = [RandomGhost(i+1) for i in range(state.getNumAgents())]
                else: ghosts = [DirectionalGhost(i+1) for i in range(state.getNumAgents())]
                for c_turn in range(self.depth_simulation):
                    while agent_index < state.getNumAgents():
                        if state.isWin() or state.isLose(): return state.isWin(), state.getScore()
                        if agent_index == 0: state, action = q_learning(state)
                        else:
                            ghost = ghosts[agent_index-1]
                            state = state.generateSuccessor(agent_index, ghost.getAction(state))
                        agent_index += 1
                    agent_index = 0
                return heuristic_fn(state)
            
        def state_finder(c_node, target_search, depth=0):
            state_fnd = None
            if c_node.agent_index == 0 and depth > 0:
                if target_search == c_node.state: state_fnd = c_node
            else:
                for c_child in c_node.children:
                    state_fnd = state_finder(c_child, target_search, 1)
                    if state_fnd is not None: break
            return state_fnd
        
        
        

        # Monte Carlo tree search  

        start_time = time.time()
        if MonteCarloTreeSearchAgent.c_tree is not None and self.structure_reuse: tree = state_finder(MonteCarloTreeSearchAgent.c_tree, gameState, 0)
        else: tree = None
        if tree is None: tree = Node(gameState, action=None, parent=None)
        else: tree.parent = None
        tree = Node(gameState, action=None, parent=None)
        cntr = 0
        cntr_bored = 0
        c_rate_win = 1
        act_last = -1
        
        while cntr < self.allowed_steps:
            leaf = leaf_select(tree)
            leaf_expand(leaf)
            if leaf.children:
                c = random.choice(leaf.children)
                res = simulate_rand_act(c, c.agent_index+1)
                backpropagation(res, c)
            else:
                res = leaf.state.isWin(), leaf.state.getScore()
                backpropagation(res, leaf)
            cntr +=1
            if self.near_stop:
                if self.bool_panics:
                    c_rate_win = ((1 - self.sim_relevance) * c_rate_win) + (self.sim_relevance * res[0])
                    if c_rate_win <= self.panic_rate: cntr_bored = -1
                c_act_top = tree.find_action(b_child_algorithm=self.choose_algo)
                if c_act_top == act_last:
                    cntr_bored += 1
                    if cntr_bored >= self.tillbored_steps: break
                else:
                    act_last = c_act_top
                    cntr_bored = 0

        Node.id_node = 0
        MonteCarloTreeSearchAgent.c_tree = tree
        act = tree.find_action(b_child_algorithm=self.choose_algo)
        #end-time
        end_time = time.time()
        #finding the parameters  
        MultiAgentSearchAgent.depth_of_tree.append(find_depth_max(tree))
        MultiAgentSearchAgent.number_of_nodes.append(self.current_number_of_nodes)
        MultiAgentSearchAgent.time_per_moves.append(end_time - start_time)
        self.current_number_of_nodes = 0
        #return the relevant action
        return act            
           
