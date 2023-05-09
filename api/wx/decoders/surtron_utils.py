import re 
import os
from datetime import timedelta
from tempestas_api import settings

def clean_line(string):
    count = 0
    while string[0] == '\t':
        string = string[1:]
        count += 1
       
    string = string[:-1]
    
    return (count, string)

def add(i):
    i[0]+=1
    
def val(i):
    return i[0]

def get_var(line):
    key, var = re.split('=',line)
    
    key = key.strip()
    var = var.strip()
    var = var.strip('"')
    
    return key, var

def get_vars(tab, i, lines):
    dictionary = {}
    
    while True: 
        count, line = lines[val(i)]         
        
        if count <= tab:
            break
        if count > tab+1:
            raise('Error in get_vars')
            break
            
        key, var = get_var(line)            
        dictionary[key] = var
        
        add(i)
    return dictionary

def get_key(line):
    if line[0] == '<':
        if line[-1] == '>':
            key = line[1:-1]
        else:
            key = line[1:]
    else:
        key = None
    
    return key

def get_element(line, tab, i, lines):
    if line[0] == '<':
        if line[1] == '/':
            add(i)
            element = None        
        elif line[-1] == '>':
            if 'List' in line:
                add(i)
                element = get_list(tab+1, i, lines)
            else:
                add(i)
                element = get_dict(tab+1, i, lines)
        else:               
            add(i)
            element = get_vars(tab+1, i, lines)
    else:
        add(i)
        element = None
    
    return element

def get_list(tab, i, lines):
    lst = []
    
    while True:
        count, line = lines[val(i)]        
        
        if count <= tab:
            break
            
        if count > tab+1:
            print('Error in get_list')
            break 

        
        element = get_element(line, tab, i, lines)
        if element is not None:
            lst.append(element)
            
    return lst

def get_dict(tab, i, lines):
    dictionary = {}
    
    while True:
        if val(i) >= len(lines):
            break

        print('Trying')
        count, line = lines[val(i)]
        print('OK')
                
        if count <= tab:
            break
            
        if count > tab+1:
            print('Error in get_dict')
            break         
        
        key = get_key(line)
        
        print('Element')
        element = get_element(line, tab, i, lines)
        
        if key is not None and element is not None:
            dictionary[key]=element
                
    return dictionary

def ssf_to_dict(config_data):
    # with open(file_path, 'r') as f:
        # lines = f.readlines()

    # lines = file_path.open('r').readlines()
    # lines = [clean_line(line) for line in lines]           
    # dictionary = get_dict(-1, [0], lines)


    lines = config_data.splitlines()
    lines = [clean_line(line+'\n') for line in lines]

    dictionary = get_dict(-1, [0], lines)
    
    return dictionary

############################################

def is_integer(var):
    if isinstance(var, int):
        return True
    if isinstance(var, float):
        return var.is_integer()
    if isinstance(var, str):
        return var.isdigit()    
    return False

def time_string_to_minutes(time_string):
    time_list = time_string.split(':')    
    time_list = list(map(int, time_list))
        
    time_delta = timedelta(hours=time_list[0], minutes=time_list[1], seconds=time_list[2])
    
    seconds = time_delta.total_seconds()
    minutes = seconds/60
    
    if is_integer(minutes):
        return int(minutes)    
    return minutes
    
def get_Offset(module_SelfTimed):
    Offset = module_SelfTimed['Offset'].split('#', 1)[0]
    minutes = time_string_to_minutes(Offset)
    return minutes

def get_Interval(module_SelfTimed):
    Interval = module_SelfTimed['Interval'].split('#', 1)[0]
    minutes = time_string_to_minutes(Interval)
    return minutes

def get_Label(module_SelfTimed):
    Label = module_SelfTimed['Label'].split('#', 1)[0]
    return Label

def get_NumVals(module_SelfTimed):
    NumVals = module_SelfTimed['NumVals'].split('#', 1)[0]        
    if is_integer(NumVals):
        return int(NumVals)
    return NumVals

def get_Divider(module_LAN):
    RightDigits = module_LAN['RightDigits'].split('#', 1)[0]        
    if is_integer(RightDigits):
        RightDigits = int(RightDigits)
    
    if RightDigits == 0:
        Divider = 0
    else:
        Divider = pow(10, RightDigits)
        Divider = int(Divider)
    
    return Divider

def get_module_elements(module_SelfTimed, module_LAN):
    Label = get_Label(module_SelfTimed)
    NumVals = get_NumVals(module_SelfTimed)
    Offset = get_Offset(module_SelfTimed)    
    Interval = get_Interval(module_SelfTimed)
    Divider = get_Divider(module_LAN)
    
    elements = []
    for i in range(NumVals):
        time_diff = Offset+i*Interval
        element = (Label, Divider, time_diff)        
        elements.append(element)
        
    return elements

def get_interval_lookup_table(modules_SelfTimed, modules_LAN):

    interval_lookup_table = {}

    for index, __ in enumerate(modules_SelfTimed):
        module_SelfTimed = modules_SelfTimed[index]
        module_LAN = modules_LAN[index]

        Label = get_Label(module_SelfTimed)
        Interval = get_Interval(module_SelfTimed) # Minutes
        Interval =  Interval*60 # Minutes to Seconds

        interval_lookup_table[Label] = Interval

    return interval_lookup_table



def get_ID_Decoder(modules_SelfTimed, modules_LAN):
    # Module List must be sorted by 'Sequence'
    ID_Decoder = []
    for index, __ in enumerate(modules_SelfTimed):

        module_SelfTimed = modules_SelfTimed[index]
        module_LAN = modules_LAN[index]

        module_elements = get_module_elements(module_SelfTimed, module_LAN)
        ID_Decoder += module_elements
    return ID_Decoder

############################################

def get_config(config_data):
    dictionary = ssf_to_dict(config_data)

    modules = [module for module in dictionary['SetupFile']['ModuleList'] if module['Name'] == 'LAN']
    modules_LAN = sorted(modules, key=lambda module: int(module['Sequence'].split('#', 1)[0]))

    modules = [module for module in dictionary['SetupFile']['ModuleList'] if module['Name'] == 'SelfTimed']
    modules_SelfTimed = sorted(modules, key=lambda module: int(module['Sequence'].split('#', 1)[0])) 

    ID_Decoder = get_ID_Decoder(modules_SelfTimed, modules_LAN)
    interval_lookup_table = get_interval_lookup_table(modules_SelfTimed, modules_LAN)
    return ID_Decoder, interval_lookup_table