# Import Modules
from random import randint
from numpy import absolute

# Input Values
red = 0.76
green = 0.32
blue = 0.65

# Training Iterations
t = 100
iteration = 0
iterationgroup = 0
r = randint(0, 256)
g = randint(0, 256)
b = randint(0, 256)
h = 0

# Convert RGB to Hue Function
def rgbtohue(r, g, b):
    maxcolor = max(r, g, b)
    mincolor = min(r, g, b)
    rcolor = (maxcolor-r) / (maxcolor-mincolor)
    gcolor = (maxcolor-g) / (maxcolor-mincolor)
    bcolor = (maxcolor-b) / (maxcolor-mincolor)
    if r is maxcolor:
        h = bcolor-gcolor
    elif g is maxcolor:
        h = 2+rcolor-bcolor
    else:
        h = 4+gcolor-rcolor
    h = (h/6) % 1
    return round(h*360)

# Nodes
oaaa = randint(0, 1000001)/1000000-0.5
oaab = randint(0, 1000001)/1000000-0.5
oaac = randint(0, 1000001)/1000000-0.5
oaba = randint(0, 1000001)/1000000-0.5
oabb = randint(0, 1000001)/1000000-0.5

obaa = randint(0, 1000001)/1000000-0.5
obab = randint(0, 1000001)/1000000-0.5
obac = randint(0, 1000001)/1000000-0.5
obba = randint(0, 1000001)/1000000-0.5
obbb = randint(0, 1000001)/1000000-0.5

ocaa = 0
ocab = 0
ocac = 0
ocba = 0
ocbb = 0

# Training
print("Training...")

while iteration < t:
    # Define Iteration Training Variables
    r = randint(0, 256)
    g = randint(0, 256)
    b = randint(0, 256)
    h = rgbtohue(r, g, b)
    r = r/255
    g = g/255
    b = b/255
    h = round(h/360, 3)

    # Define Current Iteration Network Biases
    oaaa = oaaa
    oaab = oaab
    oaac = oaac
    oaba = oaba
    oabb = oabb
    
    obaa = obaa
    obab = obab
    obac = obac
    obba = obaa
    obbb = obbb

    ocaa = randint(0, 1000001)/1000000-0.5
    ocab = randint(0, 1000001)/1000000-0.5
    ocac = randint(0, 1000001)/1000000-0.5
    ocba = randint(0, 1000001)/1000000-0.5
    ocbb = randint(0, 1000001)/1000000-0.5

    ba = 0
    bb = 0
    bc = 0
    bd = 0
    be = 0

    sba = 0
    sbb = 0
    sbc = 0
    sbd = 0
    sbe = 0

    # Solve Network Outputs (Linear Activation Function)
    oao = ((((((((r+g+b)/3)+oaaa)+(((r+g+b)/3)+oaab)+(((r+g+b)/3)+oaac))/3)+oaba)+((((((r+g+b)/3)+oaaa)+(((r+g+b)/3)+oaab)+(((r+g+b)/3)+oaac))/3)+oabb))/2)
    obo = ((((((((r+g+b)/3)+obaa)+(((r+g+b)/3)+obab)+(((r+g+b)/3)+obac))/3)+obba)+((((((r+g+b)/3)+obaa)+(((r+g+b)/3)+obab)+(((r+g+b)/3)+obac))/3)+obbb))/2)
    oco = ((((((((r+g+b)/3)+ocaa)+(((r+g+b)/3)+ocab)+(((r+g+b)/3)+ocac))/3)+ocba)+((((((r+g+b)/3)+ocaa)+(((r+g+b)/3)+ocab)+(((r+g+b)/3)+ocac))/3)+ocbb))/2)
    
    # Determine Best Neural Networks
    oaoq = oao-h
    oboq = obo-h
    ocoq = oco-h
    oaoabsolute = absolute(oaoq)
    oboabsolute = absolute(oboq)
    ocoabsolute = absolute(ocoq)
    fitness = ((oaoabsolute+oboabsolute+ocoabsolute)/3)*100
    worst = max(oaoabsolute, oboabsolute, ocoabsolute)
    best = min(oaoabsolute, oboabsolute, ocoabsolute)
    print("Generation Fitness:", fitness)

    # Prepare Variables Next Generation and Define Best Networks
    if best is oaoabsolute:
        ba = oaaa
        bb = oaab
        bc = oaac
        bd = oaba
        be = oabb

    if best is oboabsolute:
        ba = obaa
        bb = obab
        bc = obac
        bd = obba
        be = obbb

    if best is ocoabsolute:
        ba = ocaa
        bb = ocab
        bc = ocac
        bd = ocba
        be = ocbb

    if worst is not oaoabsolute and best is not oaoabsolute:
        sba = oaaa
        sbb = oaab
        sbc = oaac
        sbd = oaba
        sbe = oabb
    
    if worst is not oboabsolute and best is not oboabsolute:
        sba = obaa
        sbb = obab
        sbc = obac
        sbd = obba
        sbe = obbb
    
    if worst is not ocoabsolute and best is not ocoabsolute:
        sba = ocaa
        sbb = ocab
        sbc = ocac
        sbd = ocba
        sbe = ocbb
    
    oaaa = ba
    oaab = bb
    oaac = bc
    oaba = bd
    oabb = be

    obaa = sba
    obab = sbb
    obac = sbc
    obba = sbd
    obbb = sbe

    iteration = iteration+1
    
# Print Results
print("Training Done!")
print()
print("Result:", round(oao*360),"Actual:", rgbtohue(red, green, blue))
print()
print("Debug Info:")
print("Last Generation Hue:", h, "Network A, B, and C Distance from Target:", oaoabsolute, oboabsolute, ocoabsolute)
