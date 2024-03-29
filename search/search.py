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
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())

    """
    "*** YOUR CODE HERE ***"
    """actionsList contains the actions that will be taken during the search of a goal node from the given start node"""
    actionsList = []
    """if the starting node is the goal node actionslist is returned empty"""
    if problem.isGoalState(problem.getStartState()):
        return actionsList
    visited = set([])
    visited.add(problem.getStartState())
    level = 1
    successors = problem.getSuccessors(problem.getStartState())
    """LIFO data structure stack is used in case of DFS"""
    myStack = util.Stack()
    for succ in successors:
        succList = list(succ)
        succList.append(level)
        succ = tuple(succList)
        myStack.push(succ)
    reachedLeaf = True
    while myStack.isEmpty() == False:
        nextNode = myStack.pop()
        if reachedLeaf == True:
            while len(actionsList) >= nextNode[3]:
                actionsList.pop()
        actionsList.append(nextNode[1])
        if problem.isGoalState(nextNode[0]):
            print "Goal state found"
            return actionsList
        visited.add(nextNode[0])
        successors = problem.getSuccessors(nextNode[0])
        for succ in successors:
            if succ not in myStack.list and succ[0] not in visited:
                succList = list(succ)
                succList.append(nextNode[3]+1)
                succ = tuple(succList)
                myStack.push(succ)
                readchedLeaf = False


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    actionsList = []
    if problem.isGoalState(problem.getStartState()):
        return actionsList
    visited = set([])
    frontier = list()
    visited.add(problem.getStartState())
    successors = problem.getSuccessors(problem.getStartState())
    myQueue = util.Queue()
    pathQueue = util.Queue()
    for succ in successors:
        """add all the successor nodes in the queue"""
        if succ not in frontier and succ[0] not in visited:
            pathList = list()
            pathList.insert(len(pathList),succ[1])
            pathQueue.push(pathList)
            myQueue.push(succ)
            frontier.insert(len(frontier),succ[0])
    """adding successors to the queue first and when popping them out of the queue will check if it is the goal state or not"""
    while myQueue.isEmpty() == False:
        """since a queue is a FIFO data structure nodes at the same level will be traversed first when pop is called"""
        nextNode = myQueue.pop()
        nextPath = pathQueue.pop()
        visited.add(nextNode[0])
        if problem.isGoalState(nextNode[0]):
            """if the node popped is the goal node then we will return the path to this node that is stored in a seperate queue pathQueue"""
            pathList = list(nextPath)
            return pathList
        successors = problem.getSuccessors(nextNode[0])
        for succ in successors:
            if succ[0] not in frontier and succ[0] not in visited:
                pathList = list(nextPath)
                pathList.insert(len(pathList),succ[1])
                pathQueue.push(pathList)
                myQueue.push(succ)
                frontier.insert(len(frontier),succ[0])

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    """this is same as the breadth first search, the only
    difference is that the data structure used is a
    priority queue and the factor used to decide priority is the
    cost till that node"""

    actionsList = []
    if problem.isGoalState(problem.getStartState()):
        return actionsList
    visited = set([])
    frontier = {}
    visited.add(problem.getStartState())
    successors = problem.getSuccessors(problem.getStartState())
    myQueue = util.PriorityQueue()
    pathQueue = {}
    for succ in successors:
        if succ not in frontier and succ[0] not in visited:
            pathList = list()
            pathList.insert(len(pathList),succ[1])
            pathQueue[succ[0]] = pathList
            myQueue.push(succ[0],succ[-1])
            frontier[succ[0]]=succ[-1]

    while myQueue.isEmpty() == False:
        nextNode = myQueue.pop()
        nextPath = pathQueue.pop(nextNode, None)
        costTillNow=frontier.pop(nextNode, None)
        visited.add(nextNode)
        if problem.isGoalState(nextNode):
            pathList = list(nextPath)
            return pathList
        successors = problem.getSuccessors(nextNode)
        for succ in successors:
            if succ[0] not in frontier and succ[0] not in visited:
                pathList = list(nextPath)
                pathList.insert(len(pathList),succ[1])
                pathQueue[succ[0]] = pathList
                frontier[succ[0]]=succ[-1]+costTillNow
                myQueue.push(succ[0],frontier[succ[0]])
            elif succ[0] in frontier and frontier[succ[0]] > succ[-1]+costTillNow:
                frontier.pop(succ[0], None)
                pathQueue.pop(succ[0], None)
                pathList = list(nextPath)
                pathList.insert(len(pathList),succ[1])
                pathQueue[succ[0]] = pathList
                myQueue.update(succ[0],succ[-1])
                frontier[succ[0]]=succ[-1]


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    """this is same as the UCS, the only
    difference is that the factor used to decide priority is the heuristic + 
    cost till that node"""

    actionsList = []
    if problem.isGoalState(problem.getStartState()):
        return actionsList
    visited = set([])
    frontier = {}
    visited.add(problem.getStartState())
    successors = problem.getSuccessors(problem.getStartState())
    myQueue = util.PriorityQueue()
    pathQueue = {}
    for succ in successors:
        if succ not in frontier and succ[0] not in visited:
            combinedCost = succ[-1]+heuristic(succ[0],problem)
            pathList = list()
            pathList.insert(len(pathList),succ[1])
            pathQueue[succ[0]] = pathList
            myQueue.push(succ[0],combinedCost)
            frontier[succ[0]]=combinedCost

    while myQueue.isEmpty() == False:
        nextNode = myQueue.pop()
        nextPath = pathQueue.pop(nextNode, None)
        costForPath = frontier.pop(nextNode,None)-heuristic(nextNode,problem)
        visited.add(nextNode)
        if problem.isGoalState(nextNode):
            pathList = list(nextPath)
            return pathList
        successors = problem.getSuccessors(nextNode)
        for succ in successors:
            combinedCost = succ[-1]+costForPath+heuristic(succ[0],problem)
            if succ[0] not in frontier and succ[0] not in visited:
                pathList = list(nextPath)
                pathList.insert(len(pathList),succ[1])
                pathQueue[succ[0]] = pathList
                myQueue.push(succ[0],combinedCost)
                frontier[succ[0]]=combinedCost
            elif succ[0] in frontier and frontier[succ[0]] > combinedCost:
                frontier.pop(succ[0], None)
                pathQueue.pop(succ[0], None)
                pathList = list(nextPath)
                pathList.insert(len(pathList),succ[1])
                pathQueue[succ[0]] = pathList
                myQueue.update(succ[0],combinedCost)
                frontier[succ[0]]=combinedCost

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
