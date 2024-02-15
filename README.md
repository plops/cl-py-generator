# Introduction

This program is a Common Lisp tool to convert s-expressions into Python.

## Supported Forms

I used Chat GPT 4 to get started with the following documentation. Eventually, I have to go through this and fix the mistakes it made.

The prompt I used is:

```
create documentation for the forms that are supported by the selected code.  

the output shall be like:    
tuple .. Python tuple. Example: (tuple 1 2) => (1,2,)   
paren .. Comma separated values in parentheses. Example: (paren 1 2) => (1, 2)   
ntuple .. comma separeted values without parentheses. This can be used as an assignment target. Example: (setf (ntuple a b) (fun)) => a, b = fun()  
<source code of the big case statement in emit-py>
```

I shall try again and tell Chat GPT 4 that the code is Common Lisp and we are dealing with a s-expression based syntax for Python. Maybe it will do a better job.
   
**Tuple**  
- **Python tuple**: A finite ordered list of elements. Tuples are immutable and typically used to store collections of heterogeneous items.  
- **Example**: `(tuple 1 2)` => `(1, 2,)`  
   
**Paren**  
- **Comma separated values in parentheses**: A sequence of values separated by commas and enclosed in parentheses. Used to define ordered collections of items.  
- **Example**: `(paren 1 2)` => `(1, 2)`  
   
**NTuple**  
- **Comma separated values without parentheses**: A sequence of values separated by commas, not enclosed in parentheses. This format is commonly used for unpacking iterables into variables.  
- **Example**: `(setf (ntuple a b) (fun))` => `a, b = fun()`  
   
**List**  
- **Python list**: A mutable sequence, typically used to store collections of homogenous items.  
- **Example**: `(list 1 2)` => `[1, 2]`  
   
**Curly**  
- **Set or Dictionary initializer**: Initializes a Python set or dictionary using curly braces.  
- **Example**: `(curly 1 2)` => `{1, 2}` for a set or `{key1: value1, key2: value2}` for a dictionary if used with pairs.  
   
**Dict**  
- **Python dictionary**: A mutable mapping of unique keys to values. Dictionaries are indexed by keys.  
- **Example**: `(dict (a 1) (b 2))` => `{'a': 1, 'b': 2}`  
   
**Dictionary**  
- **Python dictionary using 'dict' constructor**: Similar to 'Dict', but uses the 'dict()' constructor to build the dictionary.  
- **Example**: `(dictionary (a 1) (b 2))` => `dict(a=1, b=2)`  
   
**Indent**  
- **Indented block**: Adds indentation to a block of code, commonly used inside code blocks like loops and conditionals to indicate scope.  
- **Example**: `(indent level code)` => `"    code"` (where 'level' is a number indicating the depth of indentation)  
   
**Do**  
- **Execute a series of statements**: Allows multiple statements to be executed in a sequence within the scope where 'do' appears.  
- **Example**: `(do (stmt1) (stmt2))` => executes 'stmt1' and then 'stmt2'  
   
**Class**  
- **Python class definition**: Defines a new class with given parents and body of methods or attributes.  
- **Example**: `(class MyClass (Parent1 Parent2) (def __init__ (self arg)))` => defines a class 'MyClass' inheriting from 'Parent1' and 'Parent2' with an '__init__' method  
   
**Do0**  
- **Execute a series of statements starting with a special statement**: Similar to 'do', but ensures a specific statement is executed first.  
- **Example**: `(do0 (first-statement) (stmt1) (stmt2))` => ensures 'first-statement' runs before 'stmt1' and 'stmt2'  
   
**Cell**  
- **Grouping related statements**: Used to group a series of related statements, commonly used for organizational purposes in the code.  
- **Example**: `(cell (stmt1) (stmt2))` => groups 'stmt1' and 'stmt2' logically together  
   
**Export**  
- **Mark code for export**: Indicates that the enclosed code should be available for use outside of its defining module.  
- **Example**: `(export (def func (arg1)))` => indicates that function 'func' should be exported  
   
**Space**  
- **Separate values by space**: Concatenates a sequence of strings or representations of objects, separating them with a space.  
- **Example**: `(space "Hello" "World")` => `"Hello World"`  
   
**Lambda**  
- **Anonymous function**: Defines an anonymous function (a lambda function) with the specified arguments and body.  
- **Example**: `(lambda (x) (add x 1))` => `lambda x: x + 1`  
   
**Def**  
- **Function definition**: Defines a function with a given name, parameters, and body.  
- **Example**: `(def func (x y) (add x y))` => `def func(x, y): return x + y`  
   
**Setf**  
- **Assignment statement**: Used to assign values to variables.  
- **Example**: `(setf (a b) (c d))` => `a = c; b = d`  

**Incf**  
- **Increment field**: Increases the value of a variable by a specified amount.  
- **Example**: `(incf a 2)` => `a += 2`  
   
**Decf**  
- **Decrement field**: Decreases the value of a variable by a specified amount.  
- **Example**: `(decf a 2)` => `a -= 2`  
   
**Aref**  
- **Array reference**: Retrieves an element from an array or list at a given index.  
- **Example**: `(aref array-name 1)` => `array_name[1]`  
   
**Slice**  
- **Slicing**: Accesses a range of elements from a sequence.  
- **Example**: `(slice 1 2)` => `[1:2]`  
   
**Dot**  
- **Attribute access**: Accesses an attribute of an object.  
- **Example**: `(dot obj attr)` => `obj.attr`  
   
**Plus (+)**  
- **Addition**: Calculates the sum of the given arguments.  
- **Example**: `(+ 1 2)` => `(1) + (2)`  
   
**Minus (-)**  
- **Subtraction**: Calculates the difference between the given arguments.  
- **Example**: `(- 5 3)` => `(5) - (3)`  
   
**Multiply (*)**  
- **Multiplication**: Calculates the product of the given arguments.  
- **Example**: `(* 2 3)` => `(2) * (3)`  
   
**At (@)**  
- **Decorator or matrix multiplication**: Used as a decorator indicator or for matrix multiplication in later versions of Python.  
- **Example**: `(@ decorator)` => `@decorator`; `(@ matrix1 matrix2)` => `(matrix1) @ (matrix2)`  
   
**Equals (==)**  
- **Equality comparison**: Compares two values for equality.  
- **Example**: `(== a b)` => `(a) == (b)`  
   
**Left shift (<<)**  
- **Bitwise left shift**: Shifts the bits of a number to the left by a specified number of positions.  
- **Example**: `(<< a 2)` => `(a) << (2)`  
   
**Not Equals (!=)**  
- **Inequality comparison**: Compares two values for inequality.  
- **Example**: `(!= a b)` => `(a) != (b)`  
   
**Less Than (<)**  
- **Less than comparison**: Checks if one value is less than another.  
- **Example**: `(< a b)` => `(a) < (b)`  
   
**Greater Than (>)**  
- **Greater than comparison**: Checks if one value is greater than another.  
- **Example**: `(> a b)` => `(a) > (b)`  
   
**Less Than or Equals (<=)**  
- **Less than or equal to comparison**: Checks if one value is less than or equal to another.  
- **Example**: `(<= a b)` => `(a) <= (b)`  
   
**Greater Than or Equals (>=)**  
- **Greater than or equal to comparison**: Checks if one value is greater than or equal to another.  
- **Example**: `(>= a b)` => `(a) >= (b)`  
   
**Right shift (>>)**  
- **Bitwise right shift**: Shifts the bits of a number to the right by a specified number of positions.  
- **Example**: `(>> a 2)` => `(a) >> (2)`  
   
**Divide (/)**  
- **Division**: Divides one number by another.  
- **Example**: `(/ 10 2)` => `((10)/(2))`  
   
**Exponentiation (**)**  
- **Exponentiation**: Raises a number to the power of another number.  
- **Example**: `(** 2 3)` => `((2)**(3))`  
   
**Floor division (//)**  
- **Integer division**: Divides one number by another, rounding down to the nearest whole number.  
- **Example**: `(// 10 3)` => `((10)//(3))`  
   
**Modulo (%)**  
- **Remainder operation**: Calculates the remainder of a division.  
- **Example**: `(% 10 3)` => `((10)%(3))`  
   
**And**  
- **Logical AND**: Performs a logical AND operation between multiple conditions.  
- **Example**: `(and cond1 cond2)` => `(cond1) and (cond2)`  
   
**Bitwise AND (&)**  
- **Bitwise AND**: Performs a bitwise AND operation between multiple numbers.  
- **Example**: `(& a b)` => `(a) & (b)`  
   
**Logand**  
- **Bitwise Logical AND**: Another notation for performing a bitwise AND operation.  
- **Example**: `(logand a b)` => `(a) & (b)`  
   
**Logxor**  
- **Bitwise Exclusive OR (XOR)**: Performs a bitwise XOR operation between multiple numbers.  
- **Example**: `(logxor a b)` => `(a) ^ (b)`  
   
**Or**  
- **Logical OR**: Performs a logical OR operation between multiple conditions.  
- **Example**: `(or cond1 cond2)` => `(cond1) or (cond2)`  
   
**Logior**  
- **Bitwise Inclusive OR**: Performs a bitwise OR operation between multiple numbers.  
- **Example**: `(logior a b)` => `(a) | (b)`  
   
The provided code snippet appears to map these forms to a Lisp-like syntax, which suggests it is part of a system that translates a domain-specific language into Python code. The names of the forms (`incf`, `aref`, `slice`, etc.) and their behaviors are indicative of a Lisp or Scheme dialect, which typically uses prefix notation and parentheses to structure code. The documentation provided describes each form's general behavior and provides a Python equivalent to illustrate the translation result.