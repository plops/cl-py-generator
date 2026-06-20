# Transpiler Patterns registered from Examples

This file tracks the patterns found in high-numbered example files that have been integrated into the transpiler unit tests.

| Pattern | Lisp Form | Python Output | Reference File |
|---|---|---|---|
| Unary Minus / Division | `(- x)`, `(/ x)` | `-x`, `1.0/x` | `example/171_casadi/gen02.lisp` |
| Unary Bitwise NOT | `(~ x)` | `~x` | `example/161_sqlite_embed/gen01.lisp` |
| Complex Numbers | `#c(1.0 2.0)` | `1.0 + 1j * 2.0` | `py.lisp` (intrinsic support) |
| Unary Logical NOT | `(not x)` | `not x` | `example/160_udp_holepunch/gen03.lisp` |
| Raw String Call / Ident | `("list" x)` | `list(x)` | `example/149_host_videos/gen01.lisp` |
| Empty / Open Slice | `(slice)`, `(slice "" max)` | `:`, `:max` | `example/171_casadi/gen02.lisp`, `example/157_tkinter/gen01.lisp` |
| Conditional Parentheses | `(paren* * (+ a b))` | `(a + b)` | `py.lisp` (precedence rules) |
| Raw Code Insertion | `"@threaded"`, `"#!/usr/bin/env python3"` | `@threaded`, `#!/usr/bin/env python3` | `example/143_helium_gemini/gen01.lisp` |
| Loop Control | `break`, `continue` | `break`, `continue` | `example/136_tbs/gen01.lisp` |
| Yield Statements | `yield`, `(yield x)` | `yield`, `yield(x)` | `example/163_fasthtml_sse/gen01.lisp` |
| Lambda Variants | `(lambda () 42)`, `(lambda (x y) (+ x y))` | `lambda: 42`, `lambda x, y: x + y` | `example/103_co2_sensor/gen01.lisp` |
| Class Super Call | `(super)`, `(super ImageModel self)` | `super()`, `super(ImageModel, self)` | `example/130_torch_optim/gen01.lisp` |
| Chained Dot Access | `(dot model (aref weights i j) (item))` | `model.weights[i,j].item()` | `example/130_torch_optim/gen01.lisp` |
| Unpacking Assignment | `(setf (ntuple a b) (tuple 1 2))` | `a, b = (1, 2,)` | `example/144_fasthtml/gen01.lisp` |
