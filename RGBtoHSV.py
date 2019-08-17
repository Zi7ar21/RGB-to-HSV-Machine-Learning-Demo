# Import Modules
from random import randint
from numpy import absolute

# Input Values
red = 0.76
green = 0.32
blue = 0.65

# Training Iterations
t = 1000
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
oaaa = (randint(0, 1000001)/500000-1)
oaab = (randint(0, 1000001)/500000-1)
oaac = (randint(0, 1000001)/500000-1)
oaba = (randint(0, 1000001)/500000-1)
oabb = (randint(0, 1000001)/500000-1)

obaa = (randint(0, 1000001)/500000-1)
obab = (randint(0, 1000001)/500000-1)
obac = (randint(0, 1000001)/500000-1)
obba = (randint(0, 1000001)/500000-1)
obbb = (randint(0, 1000001)/500000-1)

ocaa = 0
ocab = 0
ocac = 0
ocba = 0
ocbb = 0

netamultifitness = 0
netbmultifitness = 0
netcmultifitness = 0

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

    ocaa = (randint(0, 1000001)/500000-1)
    ocab = (randint(0, 1000001)/500000-1)
    ocac = (randint(0, 1000001)/500000-1)
    ocba = (randint(0, 1000001)/500000-1)
    ocbb = (randint(0, 1000001)/500000-1)

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

    bnetmultifitness = 0
    sbnetmultifitness = 0

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
    netamultifitness = netamultifitness+oaoabsolute
    netbmultifitness = netbmultifitness+oboabsolute
    netcmultifitness = netcmultifitness+ocoabsolute
    best = min(netamultifitness, netbmultifitness, netcmultifitness)
    worst = max(netamultifitness, netbmultifitness, netbmultifitness)
    fitness = ((oaoabsolute+oboabsolute+ocoabsolute)/3)*100

    print("Generation Fitness:", fitness)

    # Prepare Variables Next Generation and Define Best Networks
    if best is netamultifitness:
        ba = oaaa
        bb = oaab
        bc = oaac
        bd = oaba
        be = oabb
        bnetmultifitness = netamultifitness

    if best is netbmultifitness:
        ba = obaa
        bb = obab
        bc = obac
        bd = obba
        be = obbb
        bnetmultifitness = netbmultifitness

    if best is netcmultifitness:
        ba = ocaa
        bb = ocab
        bc = ocac
        bd = ocba
        be = ocbb
        bnetmultifitness = netcmultifitness

    if worst is not netamultifitness and best is not netamultifitness:
        sba = oaaa
        sbb = oaab
        sbc = oaac
        sbd = oaba
        sbe = oabb
        sbnetmultifitness = netamultifitness
    
    if worst is not netbmultifitness and best is not netbmultifitness:
        sba = obaa
        sbb = obab
        sbc = obac
        sbd = obba
        sbe = obbb
        sbnetmultifitness = netbmultifitness
    
    if worst is not netcmultifitness and best is not netcmultifitness:
        sba = ocaa
        sbb = ocab
        sbc = ocac
        sbd = ocba
        sbe = ocbb
        sbnetmultifitness = netcmultifitness
    
    if worst is netamultifitness:
        netamultifitness = 0
    
    if worst is netbmultifitness:
        netbmultifitness = 0
    
    if worst is netcmultifitness:
        netcmultifitness = 0
    
    oaaa = ba
    oaab = bb
    oaac = bc
    oaba = bd
    oabb = be
    netamultifitness = bnetmultifitness

    obaa = sba
    obab = sbb
    obac = sbc
    obba = sbd
    obbb = sbe
    netbmultifitness = sbnetmultifitness

    netcmultifitness = 0

    iteration = iteration+1
    
# Print Results
print("Training Done!")
print()
print("Result:", round(oao*360),"Actual:", rgbtohue(red, green, blue))
print()
print("Debug Info:")
print("Last Generation Hue:", h, "Network A, B, and C Distance from Target:", oaoabsolute, oboabsolute, ocoabsolute, "Best of Each:", netamultifitness, netbmultifitness, netcmultifitness)
