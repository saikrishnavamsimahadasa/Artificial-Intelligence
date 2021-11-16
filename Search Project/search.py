# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

#for back tracking in search algorithms
class Node:
    def __init__(self,position,actions,pathcost=0,heuristicfunc=0):
        self.position = position
        self.actions = actions
        self.pathcost = pathcost
        self.heuristicfunc = heuristicfunc
        
        
def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    
    
    # closed_set = []
    # fringe = util.Stack()
    # actions = []
    # node_dict = {}
    
    # root_node = problem.getStartState()
    # fringe.push(root_node)
    
    # while True:
    #     if len(fringe.list) == 0:
    #         break
    #     last_node = fringe.list.pop()
    #     if problem.isGoalState(last_node):
    #         return actions
    #     if last_node not in closed_set:
    #         closed_set.append(last_node)
    #         print(closed_set)
    #         if last_node in node_dict:
    #             current_action = node_dict.pop(last_node)
    #             actions.append(current_action)
    #             print(actions)
    #         lastnode_successors = problem.getSuccessors(last_node)
    #         if len(lastnode_successors) != 0:
    #             for i in range(0,len(lastnode_successors)):
    #                 fringe.push(lastnode_successors[i][0])
    #                 node_dict[lastnode_successors[i][0]] = lastnode_successors[i][1]
        # else:
        #     lastnode_successors = problem.getSuccessors(last_node)
        #     actions.append(lastnode_successors[i][1])
    # return actions
    
    
    fringe = util.Stack()
    closed_set = []
    root_node = problem.getStartState()

    fringe.push(Node(root_node,[]))

    while not fringe.isEmpty():
        current_node = fringe.pop()
        if problem.isGoalState(current_node.position):
            return current_node.actions
        if current_node.position not in closed_set:
            closed_set.append(current_node.position)
            successors = problem.getSuccessors(current_node.position)
            for i in successors:
                #print(current_node.position)
                #print(i)
                upd_actions = list(current_node.actions)
                #print(upd_actions)
                upd_actions.append(i[1])
                #print(upd_actions)
                fringe.push(Node(i[0], upd_actions))
    return current_node.actions

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    
    fringe = util.Queue()
    closed_set = []
    root_node = problem.getStartState()

    fringe.push(Node(root_node,[]))

    while not fringe.isEmpty():
        current_node = fringe.pop()
        if problem.isGoalState(current_node.position):
            return current_node.actions
        if current_node.position not in closed_set:
            closed_set.append(current_node.position)
            successors = problem.getSuccessors(current_node.position)
            for i in successors:
                #print(current_node.position)
                #print(i)
                upd_actions = list(current_node.actions)
                #print(upd_actions)
                upd_actions.append(i[1])
                #print(upd_actions)
                fringe.push(Node(i[0],upd_actions))
    return current_node.actions

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    
    fringe = util.PriorityQueue()
    closed_set = []
    root_node = problem.getStartState()

    fringe.push(Node(root_node,[]), 0)

    while not fringe.isEmpty():
        current_node = fringe.pop()
        if problem.isGoalState(current_node.position):
            return current_node.actions
        if current_node.position not in closed_set:
            closed_set.append(current_node.position)
            successors = problem.getSuccessors(current_node.position)
            for i in successors:
                #print(current_node.position)
                #print(i)
                upd_actions = list(current_node.actions)
                #print(upd_actions)
                upd_actions.append(i[1])
                #print(upd_actions)
                fringe.update(Node(i[0],upd_actions, i[2] + current_node.pathcost),  i[2] + current_node.pathcost)
    return current_node.actions

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    fringe = util.PriorityQueue()
    closed_set = []
    root_node = problem.getStartState()
    #print(heuristic(root_node, problem))
    fringe.push(Node(root_node,[],0,heuristic(root_node, problem)), 0)

    while not fringe.isEmpty():
        current_node = fringe.pop()
        if problem.isGoalState(current_node.position):
            return current_node.actions
        if current_node.position not in closed_set:
            closed_set.append(current_node.position)
            successors = problem.getSuccessors(current_node.position)
            for i in successors:
                #print(current_node.position)
                #print(i)
                upd_actions = list(current_node.actions)
                #print(upd_actions)
                upd_actions.append(i[1])
                #print(upd_actions)
                fringe.update(Node(i[0],upd_actions, i[2] + current_node.pathcost, heuristic(i[0], problem)), i[2] + current_node.pathcost + heuristic(i[0], problem))
    return current_node.actions


        
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
