#!/usr/bin/env python3
import numpy

#constants
view = range(-2, 3)
earshot = range(-4, 5)
deadzone = 1
start = [12, 25]
numpy.random.seed(12345)
alpha = .001
dimX = 33
dimY = 2
dimR = 4
dimZ = 10
dimIn = dimX + dimY + dimR

#build array of world map
world = numpy.ones((25, 50))

'''
        [[ 0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0],
        [  0, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 0],
        [  0, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 0],
        [  0, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 0],

        [  0, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 0],
        [  0, 1, 1, 1,   1, 1, 0, 0,   0, 0, 1, 1,   1, 1, 1, 0],
        [  0, 1, 1, 1,   1, 0, 0, 0,   0, 0, 0, 1,   1, 1, 1, 0],
        [  0, 1, 1, 1,   1, 0, 0, 0,   0, 0, 0, 1,   1, 1, 1, 0],

        [  0, 1, 1, 1,   1, 0, 0, 0,   0, 0, 0, 1,   1, 1, 1, 0],
        [  0, 1, 1, 1,   1, 0, 0, 0,   0, 0, 0, 1,   1, 1, 1, 0],
        [  0, 1, 1, 1,   1, 1, 0, 0,   0, 0, 1, 1,   1, 1, 1, 0],
        [  0, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 0],
        
        [  0, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 0],
        [  0, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 0],
        [  0, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 0],
        [  0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0]]
'''

#set location data
pos = [start[0], start[1]]

#mainatin record of last input, output, reward and state
lastX = numpy.zeros((dimX))
lastY = numpy.zeros((dimY))
lastR = numpy.zeros((dimR))
lastZ = numpy.zeros((dimZ))

#initialize the weights of the network
Wi = 2 * numpy.random.rand(dimZ, dimZ+dimIn+1) -1
Wo = 2 * numpy.random.rand(dimIn, dimZ+1) -1

#convert a vector into a cardnial direction
def cardinal(Y):
    if numpy.linalg.norm(Y) < deadzone:
        return [ 0, 0]
    angle = numpy.arctan2(Y[1], Y[0])
    if -7*numpy.pi/8 <= angle < -5*numpy.pi/8:
        return [-1,-1]
    elif -5*numpy.pi/8 <= angle < -3*numpy.pi/8:
        return [ 0,-1]
    elif -3*numpy.pi/8 <= angle < -numpy.pi/8:
        return [ 1,-1]
    elif -numpy.pi/8 <= angle < numpy.pi/8:
        return [ 1, 0]
    elif numpy.pi/8 <= angle < 3*numpy.pi/8:
        return [ 1, 1]
    elif 3*numpy.pi/8 <= angle < 5*numpy.pi/8:
        return [ 0, 1]
    elif 5*numpy.pi/8 <= angle < 7*numpy.pi/8:
        return [-1, 1]
    else:
        return [-1, 0]

#render output of world
def draw():
    for x in range(len(world)):
        string = ''
        for y in range(len(world[0])):
            if pos[0] == x and pos[1] == y:
                string += '@ '
            elif world[x][y] == -1:
                string+='# '
            elif world[x][y] == -2:
                string+='* '
            elif world[x][y] == 1:
                string+='. '
            elif world[x][y] == 0:
                string+='  '
            else:
                string+=str(world[x][y])
        print('| ' + string + '|')

#activation function
def nonLinearFunction(x):
    if x>0:
        return numpy.log(1+x)
    else:
        return -numpy.log(1-x)
nlf = numpy.vectorize(nonLinearFunction)

#derivative of the activation function
def nonLinearDerivative(x):
    if x>0:
        return 1/(1+x)
    else:
        return 1/(1-x)
nld = numpy.vectorize(nonLinearDerivative)

#process an output to compute the new inputs and rewards
def process(Y):
    #allow update of position
    global pos

    #calculate movement direction
    direction = cardinal(Y)

    #move
    lastPos = pos
    newPos = [pos[0] + direction[0], pos[1] + direction[1]]
    if  0 > newPos[0] or 0 > newPos[1] or len(world) <= newPos[0] or len(world[0]) <= newPos[1]:
        newPosValue = -1
    else:
        newPosValue = world[newPos[0]][newPos[1]]
    if newPosValue >= 0:
        pos = newPos
        world[newPos[0]][newPos[1]] = 0

    ### X ###

    #get values of the world map in view
    X1 = []
    for i in view:
        for j in view:
            if  0 > pos[0]+i or 0 > pos[1]+j or len(world) <= pos[0]+i or len(world[0]) <= pos[1]+j:
                value = -1
            else:
                value = world[pos[0] + i][pos[1] + j]
            X1.append(value)

    #note nearby preditors and prey
    pred = []
    prey = []

    #search for preditors and prey in earshot
    for i in earshot:
        for j in earshot:
            if  0 > pos[0]+i or 0 > pos[1]+j or len(world) <= pos[0]+i or len(world[0]) <= pos[1]+j:
                value = -1
            else:
                value = world[pos[0] + i][pos[1] + j]
            if value == 1:
                prey.append([i, j])
            elif value == -2:
                pred.append([i, j])

    #find closest preditor in earshot
    closePred = [[0, 0], 0]
    for p in pred:
        norm = numpy.linalg.norm(p)
        if 1/norm > closePred[1]:
            closePred = [p, 1/norm]

    #find closest prey in earshot
    closePrey = [[0, 0], 0]
    for p in prey:
        norm = numpy.linalg.norm(p)
        if 1/norm > closePrey[1]:
            closePrey = [p, 1/norm]

    #record nearest preditor and prey
    X2 = [closePred[0][0], closePred[0][1], closePrey[0][0], closePrey[0][1]]

    #get actual movement
    X3 = numpy.subtract(pos, lastPos)

    #get global location
    X4 = numpy.subtract(pos, start)

    #put all inputs together
    X = numpy.concatenate((X1, X2, X3, X4))

    ### R ###

    #reward for getting food
    R1 = [newPosValue]
    
    #reward for proximity to food
    R2 = [closePrey[1], -closePred[1]]

    #punishment for trying to move too fast
    R3 = [-numpy.square(numpy.linalg.norm(Y))]

    #put all rewards together
    R = numpy.concatenate((R1, R2, R3))

    ### END ###

    #return the pair of inputs for next cycle
    return [X, R]

#accept commands for stepping speed and exiting
cmd = ''
steps = 0

#loop until command Q given
while True:
    #draw the world
    print('OUTPUT: ' + str(lastY))
    print('REWARD: ' + str(lastR))
    draw()

    #get input for commands
    cmd = input('CMD: ')

    #chech command
    if cmd == 'q':
        exit()
    elif cmd == 'w':
        print('Wi:')
        print(Wi)
        print('Wo:')
        print(Wo)
    else:
        #try to adjust step speed
        try:
            steps = int(cmd)
        except:
            pass

    #loop over number of steps before drawing output
    for i in range(steps):
        #build input vector
        In = numpy.concatenate(([1], lastX, lastY, lastR, lastZ))
        
        #calculate internal node values
        Z = nlf(Wi @ In)
        dZ = numpy.diag(nld(Wi @ In))

        #add bias value for hidden nodes
        Hid = numpy.concatenate(([1], Z))

        #calculate output predictions
        Out = nlf(Wo @ Hid)
        dOut = numpy.diag(nld(Wo @ Hid))

        #separate output into X, Y and R predictions
        preX = Out[:dimX]
        preY = Out[dimX:dimX + dimY]
        preR = Out[dimX + dimY:]

        #calculate reward gradient
        dOutdIn = dOut @ Wo[:,1:] @ dZ @ Wi[:,1:]
        dYdR = dOutdIn[dimX + dimY:, dimZ + dimX:dimZ + dimX + dimY]

        #calculate optimal output based on prediction and reward gradient
        Y = preY + alpha*numpy.sum(dYdR, 0)

        #use actual output to move and calculate inputs and reward
        [X, R] = process(Y)

        #concatinate real values for training
        Target = numpy.concatenate((X, Y, R))

        #calculate the error
        dEdWo = numpy.outer((Out-Target) @ dOut, Hid)
        dEdWi = numpy.outer((Out-Target) @ dOut @ Wo[:,1:] @ dZ, In)

        #update weigths
        Wo -= alpha*dEdWo
        Wi -= alpha*dEdWi

        #save values for next iteration
        lastX = X
        lastY = Y
        lastR = R
        lastZ = Z
