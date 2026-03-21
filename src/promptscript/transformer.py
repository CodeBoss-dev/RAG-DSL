"""Lark Transformer: parse tree -> AST dataclasses."""

from lark import Transformer as LarkTransformer, Token

from promptscript.ast_nodes import (
    Program, Declaration, Assignment, SetParam, ForLoop, IfBlock, CompileCall,
    TypeStr, TypeInt, TypeFloat, TypeBool, TypePersona, TypeInstruct, TypeContextList,
    StringLiteral, LongStringLiteral, NumberLiteral, BoolLiteral,
    Identifier, FuncCall, ListExpr,
    BinaryCondition, NotCondition, ExprCondition,
)


class PromptScriptTransformer(LarkTransformer):
    # ------------------------------------------------------------------
    # Top level
    # ------------------------------------------------------------------

    def start(self, items):
        return Program(statements=list(items))

    # ------------------------------------------------------------------
    # Type specs
    # ------------------------------------------------------------------

    def type_str(self, _):
        return TypeStr()

    def type_int(self, _):
        return TypeInt()

    def type_float(self, _):
        return TypeFloat()

    def type_bool(self, _):
        return TypeBool()

    def type_persona(self, _):
        return TypePersona()

    def type_instruct(self, _):
        return TypeInstruct()

    def type_context_list(self, _):
        return TypeContextList()

    def type_spec(self, items):
        return items[0]

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------

    def declaration(self, items):
        return Declaration(type_spec=items[0], name=str(items[1]), value=items[2])

    def assignment(self, items):
        return Assignment(name=str(items[0]), value=items[1])

    def set_param(self, items):
        return SetParam(name=str(items[0]), value=items[1])

    def for_loop(self, items):
        var = str(items[0])
        iterable = items[1]
        body = list(items[2:])
        return ForLoop(var=var, iterable=iterable, body=body)

    def if_block(self, items):
        condition = items[0]
        then_body = items[1] if len(items) > 1 else []
        else_body = items[2] if len(items) > 2 else []
        return IfBlock(condition=condition, then_body=then_body, else_body=else_body)

    def then_body(self, items):
        return list(items)

    def else_body(self, items):
        return list(items)

    def compile_call(self, items):
        args = items[0] if items and isinstance(items[0], list) else []
        return CompileCall(args=args)

    def statement(self, items):
        return items[0]

    # ------------------------------------------------------------------
    # Arg list
    # ------------------------------------------------------------------

    def arg_list(self, items):
        return list(items)

    # ------------------------------------------------------------------
    # Conditions
    # ------------------------------------------------------------------

    def binary_condition(self, items):
        return BinaryCondition(left=items[0], op=str(items[1]), right=items[2])

    def not_condition(self, items):
        return NotCondition(operand=items[0])

    def expr_condition(self, items):
        return ExprCondition(expr=items[0])

    def condition(self, items):
        return items[0]

    # ------------------------------------------------------------------
    # Expressions
    # ------------------------------------------------------------------

    def func_call_expr(self, items):
        return items[0]

    def ident_expr(self, items):
        return Identifier(name=str(items[0]))

    def string_expr(self, items):
        raw = str(items[0])
        # Strip surrounding quotes
        quote_char = raw[0]
        inner = raw[1:-1]
        # Unescape basic sequences
        inner = inner.replace(f"\\{quote_char}", quote_char)
        inner = inner.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
        return StringLiteral(value=inner)

    def long_string_expr(self, items):
        raw = str(items[0])
        # Strip triple-quote delimiters
        inner = raw[3:-3]
        return LongStringLiteral(value=inner)

    def number_expr(self, items):
        raw = str(items[0])
        val: float | int = float(raw) if ("." in raw or "e" in raw.lower()) else int(raw)
        return NumberLiteral(value=val)

    def bool_expr(self, items):
        return BoolLiteral(value=str(items[0]) == "true")

    def list_expr(self, items):
        return ListExpr(elements=list(items))

    def expression(self, items):
        return items[0]

    # ------------------------------------------------------------------
    # Function calls
    # ------------------------------------------------------------------

    def dotted_call(self, items):
        func_name = items[0]
        args = items[1] if len(items) > 1 and isinstance(items[1], list) else []
        return FuncCall(func=func_name, args=args)

    def dotted_name(self, items):
        return ".".join(str(t) for t in items)

    # ------------------------------------------------------------------
    # Param literals
    # ------------------------------------------------------------------

    def number_literal(self, items):
        raw = str(items[0])
        val: float | int = float(raw) if ("." in raw or "e" in raw.lower()) else int(raw)
        return NumberLiteral(value=val)

    def bool_literal(self, items):
        return BoolLiteral(value=str(items[0]) == "true")

    def param_literal(self, items):
        return items[0]
