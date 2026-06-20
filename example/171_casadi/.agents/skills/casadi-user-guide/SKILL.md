---
name: casadi-user-guide
description: Reference guide for CasADi concepts and API based on the official CasADi user guide. Use this to quickly look up symbolic framework options, Opti stack, custom functions, DAE builder, and how to transpile these using cl-py-generator.
---

# CasADi User Guide Skill

This skill provides a quick reference to the core concepts and APIs of CasADi, helping you construct CasADi code and translate it into Common Lisp S-Expressions for `cl-py-generator`.

## Core Documentation Chapters

The raw documentation chapters (reStructuredText source files from the official user guide) are available in the local `references/` directory:

1. **Introduction & Concepts**: [intro.rst](file:///home/kiel/stage/cl-py-generator/example/171_casadi/.agents/skills/casadi-user-guide/references/intro.rst)
   - Describes what CasADi is (a tool for algorithmic differentiation and numerical optimization) and what it is not (a full NLP solver or modeling language in itself).
2. **Symbolic Framework**: [symbolic.rst](file:///home/kiel/stage/cl-py-generator/example/171_casadi/.agents/skills/casadi-user-guide/references/symbolic.rst)
   - SX symbolics (scalar node-based expressions).
   - MX symbolics (matrix/expression-based symbolics, more general).
   - Sparsity structures and Data Matrix (`DM`) types.
3. **Function Objects**: [function.rst](file:///home/kiel/stage/cl-py-generator/example/171_casadi/.agents/skills/casadi-user-guide/references/function.rst)
   - Creating `Function` objects from SX/MX inputs and outputs.
   - Calling, evaluation, and computing derivatives (Jacobian `jacobian`, Hessian `hessian`).
4. **C-Code Generation**: [ccode.rst](file:///home/kiel/stage/cl-py-generator/example/171_casadi/.agents/skills/casadi-user-guide/references/ccode.rst)
   - Generating C code from CasADi Functions (`f.generate()`).
   - Compiling and loading generated C code.
5. **Custom User-Defined Functions**: [custom.rst](file:///home/kiel/stage/cl-py-generator/example/171_casadi/.agents/skills/casadi-user-guide/references/custom.rst)
   - Subclassing `Callback` for custom evaluations or custom derivatives.
6. **DAE Builder**: [daebuilder.rst](file:///home/kiel/stage/cl-py-generator/example/171_casadi/.agents/skills/casadi-user-guide/references/daebuilder.rst)
   - Formulating differential-algebraic equations (DAEs) systematically.
7. **Optimal Control**: [ocp.rst](file:///home/kiel/stage/cl-py-generator/example/171_casadi/.agents/skills/casadi-user-guide/references/ocp.rst)
   - Formulating optimal control problems using direct methods (single/multiple shooting).
8. **Opti Stack**: [opti.rst](file:///home/kiel/stage/cl-py-generator/example/171_casadi/.agents/skills/casadi-user-guide/references/opti.rst)
   - The high-level helper classes for NLP formulation (Opti, variables, parameters, objective, constraints, solvers).
9. **Advanced Usage**: [usage.rst](file:///home/kiel/stage/cl-py-generator/example/171_casadi/.agents/skills/casadi-user-guide/references/usage.rst)
   - Memory management, options, custom solvers, exceptions, and configuration.

---

## CL-Py-Generator Syntax Mapping for CasADi

Below is a reference showing how to map standard CasADi API constructs from Python to `cl-py-generator` Lisp S-Expressions.

### 1. Symbolic Types

| Python CasADi Code | CL-Py-Generator S-Expression |
| :--- | :--- |
| `x = SX.sym("x")` | `(setf x (SX.sym (string "x")))` |
| `y = SX.sym("y", 5)` | `(setf y (SX.sym (string "y") 5))` |
| `Z = SX.sym("Z", 4, 2)` | `(setf Z (SX.sym (string "Z") 4 2))` |
| `u = MX.sym("u", 2, 2)` | `(setf u (MX.sym (string "u") 2 2))` |
| `B1 = SX.zeros(4, 5)` | `(setf B1 (SX.zeros 4 5))` |
| `B2 = SX(4, 5)` | `(setf B2 (SX 4 5))` |
| `B3 = SX.eye(4)` | `(setf B3 (SX.eye 4))` |

### 2. Matrix Access & Operators

| Python CasADi Code | CL-Py-Generator S-Expression |
| :--- | :--- |
| `f = x**2 + 10` | `(setf f (+ (** x 2) 10))` |
| `g = 3 * z + x` | `(setf g (+ (* 3 z) x))` |
| `y[0, 0] = x` | `(setf (aref y 0 0) x)` |
| `C = DM(2, 3)` | `(setf C (DM 2 3))` |
| `C_dense = C.full()` | `(setf C_dense (C.full))` |

### 3. Function & Derivatives

| Python CasADi Code | CL-Py-Generator S-Expression |
| :--- | :--- |
| `f = Function('f', [x, y], [f_expr])` | `(setf f (Function (string "f") (list x y) (list f_expr)))` |
| `J = jacobian(f, x)` | `(setf J (jacobian f x))` |
| `H, g = hessian(f, x)` | `(setf (ntuple H g) (hessian f x))` |

### 4. Opti Stack API

| Python CasADi Code | CL-Py-Generator S-Expression |
| :--- | :--- |
| `opti = casadi.Opti()` | `(setf opti (casadi.Opti))` |
| `x = opti.variable()` | `(setf x (opti.variable))` |
| `p = opti.parameter()` | `(setf p (opti.parameter))` |
| `opti.minimize((y - x**2)**2)` | `(opti.minimize (** (- y (** x 2)) 2))` |
| `opti.subject_to(x**2 + y**2 == 1)` | `(opti.subject_to (== (+ (** x 2) (** y 2)) 1))` |
| `opti.subject_to(opti.bounded(0, x, 1))` | `(opti.subject_to (opti.bounded 0 x 1))` |
| `opti.solver('ipopt')` | `(opti.solver (string "ipopt"))` |
| `sol = opti.solve()` | `(setf sol (opti.solve))` |
| `x_val = sol.value(x)` | `(setf x_val (sol.value x))` |

---

## Guide to Using References

When you need to look up details for a specific CasADi task:
- For symbolic variables, matrix sparsity, slicing, or math operations: Read [symbolic.rst](file:///home/kiel/stage/cl-py-generator/example/171_casadi/.agents/skills/casadi-user-guide/references/symbolic.rst).
- For formulating optimization problems with standard NLP solvers, objective functions, bounds, or solver parameters: Read [opti.rst](file:///home/kiel/stage/cl-py-generator/example/171_casadi/.agents/skills/casadi-user-guide/references/opti.rst).
- For defining custom dynamics, integration, or differential-algebraic equations: Read [daebuilder.rst](file:///home/kiel/stage/cl-py-generator/example/171_casadi/.agents/skills/casadi-user-guide/references/daebuilder.rst) and [ocp.rst](file:///home/kiel/stage/cl-py-generator/example/171_casadi/.agents/skills/casadi-user-guide/references/ocp.rst).
- For creating and calling functions, compiling to C: Read [function.rst](file:///home/kiel/stage/cl-py-generator/example/171_casadi/.agents/skills/casadi-user-guide/references/function.rst) and [ccode.rst](file:///home/kiel/stage/cl-py-generator/example/171_casadi/.agents/skills/casadi-user-guide/references/ccode.rst).
