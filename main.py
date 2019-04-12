#!/usr/bin/env python3
import numpy

#world constants
grid = 5
view = 15
viewAngle = 0.5*numpy.pi
viewInputs = 16
smell = 10
smellInputs = 16
speed = 1
speedAngle = viewAngle/2
pos = [0, 0]
angle = 0
chanceGood = .5
chanceBad = .25
world = {}

#processing constants
display = (1 + max(view, smell)//grid) * grid
viewStep = viewAngle / viewInputs
smellStep = 2*numpy.pi / smellInputs

#network constants
numpy.random.seed(12345)
alpha = .001
dimX = viewInputs + smellInputs
dimY = 2
dimR = 4
dimZ = 10
dimIn = dimX + dimY + dimR

#mainatin record of last input, output, reward and state
lastX = numpy.zeros((dimX))
lastY = numpy.zeros((dimY))
lastR = numpy.zeros((dimR))
lastZ = numpy.zeros((dimZ))

#initialize the weights of the network
Wi = 2 * numpy.random.rand(dimZ, dimZ+dimIn+1) -1
Wo = 2 * numpy.random.rand(dimIn, dimZ+1) -1

#render output of world
def draw():
    #print inputs and outputs
    print('X: ' + str(lastX))
    print('R: ' + str(lastR))
    print('Y: ' + str(lastY))
    print('Z: ' + str(lastZ))

    #find current cell
    x = pos[0]//grid
    y = pos[1]//grid

    #draw world
    for i in range(x - display, x + display, grid):
        #find row text
        row = '   '
        for j in range(y - display, y + display, grid):
            cell = getCell(i, j)
            if cell == None:
                row += '  '
            elif cell[1]:
                row += '. '
            else:
                row += '* '
        print(row)

#manage angle math
def angleSum(a1, a2=0):
    theta = a1 + a2
    while theta > numpy.pi:
        theta -= 2*numpy.pi
    while theta <= -numpy.pi:
        theta += 2*numpy.pi
    return theta

#process an output to compute the new inputs and rewards
def process(Y):
    #allow update of angle and position
    global angle
    global pos

    #update position with capped speed
    v = Y[0]
    if v > speed:
        v = speed
    elif v < -speed:
        v = -speed
    pos = [pos[0] + v*numpy.cos(angle), pos[0] + v*numpy.sin(angle)]

    #update angle with capped angular speed
    a = Y[1]
    if a > speedAngle:
        a = speedAngle
    elif a < -speedAngle:
        a = -speedAngle
    angle = angleSum(angle, a)

    ### X ###

    #calculate smell and sight inputs
    X1 = []
    X2 = []
    
    #find current cell
    x = pos[0]//grid
    y = pos[1]//grid

    #loop over cells to percieve
    for i in range(x - display, x + display, grid):
        for j in range(y - display, y + display, grid):
            #get cell information
            cell = getCell(i, j)

            #skip if cell has no information
            if cell != None:
                #calculate cell inverse distance from
                cell[0]

    '''
    #record nearest preditor and prey
    X2 = [closePred[0][0], closePred[0][1], closePrey[0][0], closePrey[0][1]]

    #get actual movement
    X3 = numpy.subtract(pos, lastPos)

    #get global location
    X4 = numpy.subtract(pos, start)

    #put all inputs together
    X = numpy.concatenate((X1, X2, X3, X4))
    '''

    ### R ###

    #reward for slow movement and rotation
    R1 = [1 - (Y[0] / speed)**2, 1 - (Y[1] / speedAngle)**2]

    '''
    #reward for proximity to food
    R2 = [closePrey[1], -closePred[1]]

    #punishment for trying to move too fast
    R3 = [-numpy.square(numpy.linalg.norm(Y))]

    #put all rewards together
    R = numpy.concatenate((R1, R2, R3))
    '''

    ### END ###

    #return the pair of inputs for next cycle
    return [X, R]

#get cell data
def getCell(x, y):
    #use world variable
    global world

    #check if cell already exists
    try:
        return world[(x,y)]
    except KeyError:
        #check if anything in cell
        cellType = numpy.random.random()
        if cellType > chanceGood + chanceBad:
            #note that nothing is in the cell
            world[(x,y)] = None
            return None

        #check if cell is good or bad
        cellGood = cellType < chanceGood

        #locate item in cell
        i = x + grid*numpy.random.random()
        j = y + grid*numpy.random.random()

        #build cell and return it
        world[(x,y)] = [(i,j), cellGood]
        return [(i,j), cellGood]

#activation function
def nonLinearFunction(x):
    if x>0:
        return numpy.log(1+x)
    else:
        return -numpy.log(1-x)

#derivative of the activation function
def nonLinearDerivative(x):
    if x>0:
        return 1/(1+x)
    else:
        return 1/(1-x)

#make functions work on matrices
nlf = numpy.vectorize(nonLinearFunction)
nld = numpy.vectorize(nonLinearDerivative)

#accept commands for stepping speed and exiting
cmd = ''
steps = 0

#loop until command Q given
while True:
    #draw outputs
    draw()
    print(world)

    #get input for commands
    cmd = input('CMD: ')

    #chech command
    if cmd == 'q':
        exit()
    elif cmd == 's':
        print('STATE: Network state goes here')
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
