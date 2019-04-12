#!/usr/bin/env python3
import numpy

#world constants
grid = 5
viewDistance = 15
viewAngle = 0.5*numpy.pi
viewInputs = 64
smellDistance = 10
smellInputs = 32
speed = 1
speedAngle = viewAngle/2
pos = [0, 0]
angle = 0
chanceGood = 0.5
chanceBad = 0.25
world = {}

#processing constants
display = (1 + max(viewDistance, smellDistance)//grid) * grid
viewStep = viewAngle / viewInputs
smellStep = 2*numpy.pi / smellInputs

#network constants
numpy.random.seed(12345)
alpha = .001
dimX = viewInputs + smellInputs
dimY = 16
dimR = 4
dimZ = dimX + dimY + dimR
dimOUT = dimX + dimY + dimR

#mainatin record of last input, output, and reward
Xl = numpy.zeros((dimX))
Yl = numpy.zeros((dimY))
Rl = numpy.zeros((dimR))

#initialize the weights of the network
Wi = 2 * numpy.random.rand(dimZ, 1 + dimX + dimY + dimR) -1
Wo = 2 * numpy.random.rand(dimOUT, dimZ+1) -1

#render output of world
def draw():
    #print inputs and outputs
    print('X1:')
    print(str(Xl[:viewInputs]))
    print('X2:')
    print(str(Xl[viewInputs:]))
    print('Y:')
    print(str(Yl))
    print('R:')
    print(str(Rl))
    print('POS:')
    print(str(pos))
    print(18*' '+'|')

    #find current cell
    x = int(pos[0]//grid) * grid
    y = int(pos[1]//grid) * grid

    #draw world
    for i in range(x - display*2, x + display*2, grid):
        #find row text
        row = '    '
        if i == x:
            row = ' -- '
        for j in range(y - display*2, y + display*2, grid):
            cell = getCell(grid*i, grid*j)
            if cell == None:
                row += '  '
            elif cell[1]:
                row += 'o '
            else:
                row += 'x '
        print(row)

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
    X1 = numpy.zeros((viewInputs))
    X2 = numpy.zeros((smellInputs))
    
    #find current cell
    x = int(pos[0]//grid) * grid
    y = int(pos[1]//grid) * grid

    #loop over cells to percieve
    for i in range(x - display, x + display, grid):
        for j in range(y - display, y + display, grid):
            #get cell information
            cell = getCell(i, j)

            #skip if cell has no information
            if cell != None:
                #get distance and vector
                v = numpy.subtract(cell[0], pos)
                d = numpy.linalg.norm(v)

                #get relative angle in range [-pi, pi] (-pi inclusive)
                a = angleSum(angle, -numpy.arctan2(v[1], v[0]))

                #calculate cell inverse distance from
                intensity = 1/d

                #get good or bad
                modifier = 2*cell[1] - 1

                #update view information if in range
                if d <= viewDistance:
                    #map angle to view input
                    index = int((a/viewAngle+0.5) * viewInputs)

                    #check if in view
                    if 0 <= index < viewInputs and intensity > abs(X1[index]):
                        #update view with closest item
                        X1[index] = modifier * intensity

                #update smell information if in range
                if d <= smellDistance:
                    #map angle to smell input
                    index = int((a+numpy.pi) / (2*numpy.pi) * smellInputs)

                    #make sure rounded correctly
                    if index < 0:
                        index = 0
                    elif index >= smellInputs:
                        index = smellInputs-1

                    #combine all smells in range
                    X2[index] += modifier * intensity
                
    #put all inputs together
    X = numpy.concatenate((X1, X2))

    ### R ###

    #reward for slow movement
    R1 = [1 - (Y[0] / speed)**2 * 0.5, 1 - (Y[1] / speedAngle)**2]

    #reward for seeing and smelling food
    R2 = [numpy.sum(X1), 0.25 * numpy.sum(X2)]

    #put all rewards together
    R = numpy.concatenate((R1, R2))

    ### END ###

    #return the pair of inputs for next cycle
    return [X, R]

#manage angle math and return number between -pi inclusive and pi exclusive
def angleSum(a1, a2=0):
    theta = a1 + a2
    while theta >= numpy.pi:
        theta -= 2*numpy.pi
    while theta < -numpy.pi:
        theta += 2*numpy.pi
    return theta

#get cell data
def getCell(x, y):
    #check cell validity
    if x%grid > 0 or y%grid > 0:
        print('(X, Y): ' + str([x, y]))

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
        ### LOOP 1 ###

        #build input vector
        IN = numpy.concatenate(([1], Xl, Yl, Rl))
        
        #calculate internal node values
        Z = nlf(Wi @ IN)
        dZ = numpy.diag(nld(Wi @ IN))

        #add bias value for hidden nodes
        HID = numpy.concatenate(([1], Z))

        #calculate output predictions
        OUT = nlf(Wo @ HID)
        dOUT = numpy.diag(nld(Wo @ HID))

        #separate output into X, Y and R predictions
        Xp = OUT[:dimX]
        Yp = OUT[dimX:dimX + dimY]
        Rp = OUT[dimX + dimY:]

        ### LOOP 2 ###

        #run above process again using preditions
        INp = numpy.concatenate(([1], Xp, Yp, Rp))
        Zp = nlf(Wi @ INp)
        dZp = numpy.diag(nld(Wi @ INp))
        HIDp = numpy.concatenate(([1], Zp))
        OUTp = nlf(Wo @ HIDp)
        dOUTp = numpy.diag(nld(Wo @ HIDp))

        #get predictions based on predictions
        Xpp = OUTp[:dimX]
        Ypp = OUTp[dimX:dimX + dimY]
        Rpp = OUTp[dimX + dimY:]

        ### OPTIMIZE OUTPUT ###

        #calculate optimal output based on prediction and reward gradient
        Y = Yp #+ alpha*numpy.sum(dYdR, 0)
        


        #custom outputs for testing
        #Y[0] = 0
        #Y[1] = .01



        #use actual output to move and calculate inputs and reward
        [X, R] = process(Y)

        ### BACK PROPAGATION ###

        #concatinate real values for training
        Target = numpy.concatenate((X, Y, R))

        #calculate the error
        dEdWo = numpy.outer((OUT-Target) @ dOUT, HID)
        dEdWi = numpy.outer((OUT-Target) @ dOUT @ Wo[:,1:] @ dZ, IN)

        #update weigths
        Wo -= alpha*dEdWo
        Wi -= alpha*dEdWi

        #save values for next iteration
        Xl = X
        Yl = Y
        Rl = R
