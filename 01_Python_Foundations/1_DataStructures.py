data = [1,2,3]
clean = []
for x in data:
    if x>0:
        clean.append(x)
print(clean)



clean1 = [x for x in data if x>0]
print(clean1)



def normalize(values):
    max_value = max(values)
    return [v/max_value for v in values]
print(normalize(data))



sensor = [{'id':'T1','rpm':1200,'valid':True},
          {'id':'T2','rpm':1000,'valid':False},
          {'id':'T3','rpm':1500,'valid':True}]

def filter_valid(sample):
    return [s for s in sample if s['valid']]
print(filter_valid(sensor),sep="space")

def extract(sample):
    return [s['rpm'] for s in sample if s['valid']]
print(extract(sensor))

def average(values):
    return sum(values)/len(values) if values else None

avgRpm = average(extract(sensor))
print(avgRpm)



val=input("Enter a no. : ")
def avg(value):
    if not value:
        raise ValueError("Empty value provided")
    return int(value)/2
try:
    result = avg(val)
    print("Result:", result)
except ValueError as e:
    print("Error:", e)

def safe_avg(value):
    try:
        return avg(value)
    except ValueError:
        return None
result = safe_avg(val)
print("Safe Result:", result)
    

def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return "File not found."
content = read_file("demo.txt")
print(content)


def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return "Cannot divide by zero."
result = divide(10, 0)
print(result)


def process_data(data):
    if not data:
        raise ValueError("No data provided")
    return [d * 2 for d in data]
try:
    processed = process_data([])
    print(processed)    
except ValueError as e:
    print("Error:", e)


def safe_process(data):
    try:
        return process_data(data)
    except ValueError:
        return []
result = safe_process([])
print("Safe Processed Data:", result)

def fetch_data(source):
    if source != "valid_source":
        raise ConnectionError("Failed to connect to data source")
    return [1, 2, 3]
try:
    data = fetch_data("invalid_source")
    print(data) 
except ConnectionError as e:
    print("Error:", e)


def safe_fetch(source):
    try:
        return fetch_data(source)
    except ConnectionError:
        return []
data = safe_fetch("invalid_source")
print("Safe Fetched Data:", data)

def calculate_mean(values):
    if not values:
        raise ValueError("Empty list provided")
    return sum(values) / len(values)
try:
    mean = calculate_mean([])
    print(mean)
except ValueError as e:
    print("Error:", e)

def safe_calculate(values):
    try:
        return calculate_mean(values)
    except ValueError:
        return 0
mean = safe_calculate([])
print("Safe Mean:", mean)

def open_config(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return "Config file not found." 
config = open_config("config.txt")
print(config)

def safe_open(file_path):
    try:
        return open_config(file_path)
    except FileNotFoundError:
        return ""   
config = safe_open("config.txt")
print("Safe Config:", config)


assert 2 + 2 == 4, "Math is broken!"
assert len([1,2,3]) == 3, "List length mismatch!"
assert all(x > 0 for x in [1,2,3]), "Not all values are positive!"
assert isinstance("hello", str), "Not a string!"
#assert 5 > 10, "This will raise an AssertionError"
print("All assertions passed!")



def check_positive(value):
    assert value > 0, "Value must be positive"
    return True
print(check_positive(10))
#print(check_positive(-5))  # This will raise an AssertionError
def check_non_empty(lst):
    assert len(lst) > 0, "List must not be empty"
    return True
print(check_non_empty([1,2,3]))
#print(check_non_empty([]))  # This will raise an AssertionError

def validate_string(s):
    assert isinstance(s, str), "Input must be a string"
    assert len(s) > 0, "String must not be empty"
    return True
print(validate_string("hello"))
#print(validate_string(""))  # This will raise an AssertionError

def validate_number(n):
    assert isinstance(n, (int, float)), "Input must be a number"
    assert n >= 0, "Number must be non-negative"
    return True
print(validate_number(10))
#print(validate_number(-5))  # This will raise an AssertionError
def check_list_elements(lst):
    assert all(isinstance(x, int) for x in lst), "All elements must be integers"
    return True
