import argparse
import pathlib
import sys
from io import StringIO

from sly import Parser

from uc.uc_ast import (
    ID,
    ArrayDecl,
    ArrayRef,
    Assert,
    Assignment,
    BinaryOp,
    Break,
    Compound,
    Constant,
    Decl,
    DeclList,
    EmptyStatement,
    ExprList,
    For,
    FuncCall,
    FuncDecl,
    FuncDef,
    GlobalDecl,
    If,
    InitList,
    ParamList,
    Print,
    Program,
    Read,
    Return,
    Type,
    UnaryOp,
    VarDecl,
    While,
)
from uc.uc_lexer import UCLexer

ENABLE_STDOUT_DEBUG = False


def debug_print(msg):
    if ENABLE_STDOUT_DEBUG:
        print(msg, file=sys.stdout)


class Coord:
    """Coordinates of a syntactic element. Consists of:
    - Line number
    - (optional) column number, for the Lexer
    """

    __slots__ = ("line", "column")

    def __init__(self, line, column=None):
        self.line = line
        self.column = column

    def __str__(self):
        if self.line and self.column is not None:
            coord_str = "@ %s:%s" % (self.line, self.column)
        elif self.line:
            coord_str = "@ %s" % (self.line)
        else:
            coord_str = ""
        return coord_str


class ParserLogger:
    """Logger Class used to log messages about the parser in a text stream.
    NOTE: This class overrides the default SlyLogger class
    """

    def __init__(self):
        self.stream = StringIO()

    @property
    def text(self):
        return self.stream.getvalue()

    def debug(self, msg, *args, **kwargs):
        self.stream.write((msg % args) + "\n")

    info = debug

    def warning(self, msg, *args, **kwargs):
        self.stream.write("WARNING: " + (msg % args) + "\n")

    def error(self, msg, *args, **kwargs):
        self.stream.write("ERROR: " + (msg % args) + "\n")

    critical = debug


class UCParser(Parser):
    tokens = UCLexer.tokens
    start = "program"
    debugfile = "parser.debug"
    log = ParserLogger()

    def __init__(self, debug=True):
        """Create a new UCParser."""
        self.debug = debug

        self.uclex = UCLexer(self._lexer_error)

        # Keeps track of the last token given to yacc (the lookahead token)
        self._last_yielded_token = None

    def parse(self, text):
        self._last_yielded_token = None
        return super().parse(self.uclex.tokenize(text))

    def _lexer_error(self, msg, line, column):
        # use stdout to match with the output in the .out test files
        print("LexerError: %s at %d:%d" % (msg, line, column), file=sys.stdout)
        sys.exit(1)

    def _parser_error(self, msg, coord=None):
        # use stdout to match with the output in the .out test files
        if coord is None:
            print("ParserError: %s" % (msg), file=sys.stdout)
        else:
            print("ParserError: %s %s" % (msg, coord), file=sys.stdout)
        sys.exit(1)

    def _token_coord(self, p):
        last_cr = self.uclex.text.rfind("\n", 0, p.index)
        if last_cr < 0:
            last_cr = -1
        column = p.index - (last_cr)
        return Coord(p.lineno, column)

    def error(self, p):
        if p:
            self._parser_error(
                "Before %s" % p.value, Coord(
                    p.lineno, self.uclex.find_tok_column(p))
            )
        else:
            self._parser_error("At the end of input (%s)" %
                               self.uclex.filename)

    # Solve ambiguity
    precedence = ( 
        ('left', 'OR', 'AND'),
        ('left', 'NE'),
        ('left', 'EQ'),
        ('left', 'GE', 'GT', 'LE', 'LT'),
        ('left', 'PLUS', 'MINUS'),
        ('left', 'TIMES', 'DIVIDE'),
        ('left', 'MOD'),
    )

    @_('')
    def empty(self, p):
        debug_print(
            f"empty (): {list(p)}")
        pass

    @_("one_or_more_global_declaration")
    def program(self, p):
        debug_print(
            f"program (global_declaration one_or_more_global_declaration): {list(p)}")
        return Program(gdecls=p[0])

    @_("unterminated_token")
    def program(self, p):
        debug_print(
            f"program (unterminated_token): {list(p)}")
        return Program(p[0])

    @_("global_declaration one_or_more_global_declaration")
    def one_or_more_global_declaration(self, p):
        debug_print(
            f"one_or_more_global_declaration (global_declaration one_or_more_global_declaration): {list(p)}")
        return [p[0]] + p[1]

    @_("global_declaration")
    def one_or_more_global_declaration(self, p):
        debug_print(
            f"one_or_more_global_declaration (global_declaration): {list(p)}")
        return [p[0]]

    @_("function_definition")
    def global_declaration(self, p):
        debug_print(
            f"global_declaration (function_definition): {list(p)}")

        return p[0]

    @_("declaration")
    def global_declaration(self, p):
        debug_print(
            f"global_declaration (declaration): {list(p)}")

        return GlobalDecl(decls=p[0].decls, coord=p[0].decls[0].coord)

    @_("type_specifier declarator compound_statement")
    def function_definition(self, p):
        debug_print(
            f"function_definition (type_specifier declarator compound_statement): {list(p)}")

        function_name = p[1].type.declname
        p[1].type.type = p[0]
        return FuncDef(type=p[0], decl=Decl(name=function_name, init=None, type=p[1]), body=p[2])
        # return FuncDef(type=p[0], decl=p[1], body=p[2])

    @_("VOID", "CHAR", "INT")
    def type_specifier(self, p):
        debug_print(
            f"type_specifier (VOID | CHAR | INT): {list(p)}")
        return Type(name=p[0], coord=self._token_coord(p))

    @_("identifier")
    def declarator(self, p):
        debug_print(
            f"declarator (identifier): {list(p)}")

        return VarDecl(declname=p[0], type=None, coord=None)


    @_("declarator LBRACKET optional_constant_expression RBRACKET")
    def declarator(self, p):
        debug_print(
            f"declarator (declarator LBRACKET optional_constant_expression RBRACKET): {list(p)}")
        # TODO: Complete
        if type(p[0]) == ArrayDecl:
            aux = p[0].dim
            p[0].dim = p[2]
            return ArrayDecl(type=p[0], dim=aux)
        return ArrayDecl(type=p[0], dim=p[2])


    @_("LPAREN declarator RPAREN")
    def declarator(self, p):
        debug_print(
            f"declarator (LPAREN declarator RPAREN): {list(p)}")

        return p[1]

    @_("declarator LPAREN optional_parameter_list RPAREN")
    def declarator(self, p):
        debug_print(
            f"declarator (declarator LPAREN optional_parameter_list RPAREN): {list(p)}")

        return FuncDecl(params=p[2], type=p[0])

    @_("constant_expression")
    def optional_constant_expression(self, p):
        debug_print(
            f"optional_constant_expression (constant_expression): {list(p)}")
        # TODO: Complete
        return p[0]

    @_("empty")
    def optional_constant_expression(self, p):
        debug_print(
            f"optional_constant_expression (empty): {list(p)}")
        # TODO: Complete
        pass

    @_("parameter_list")
    def optional_parameter_list(self, p):
        debug_print(
            f"optional_parameter_list (parameter_list): {list(p)}")
        # TODO: Complete
        return p[0]

    @_("empty")
    def optional_parameter_list(self, p):
        debug_print(
            f"optional_parameter_list (empty): {list(p)}")
        # TODO: Complete
        #return None
        pass
    
    @_("binary_expression")
    def constant_expression(self, p):
        debug_print(
            f"constant_expression (binary_expression): {list(p)}")
        # TODO: Complete
        return p[0]

    @_("unary_expression")
    def binary_expression(self, p):
        debug_print(
            f"binary_expression (unary_expression): {list(p)}")

        return p[0]

    @_("binary_expression TIMES binary_expression",
       "binary_expression DIVIDE binary_expression",
       "binary_expression MOD binary_expression",
       "binary_expression PLUS binary_expression",
       "binary_expression MINUS binary_expression",
       "binary_expression LT binary_expression",
       "binary_expression LE binary_expression",
       "binary_expression GT binary_expression",
       "binary_expression GE binary_expression",
       "binary_expression EQ binary_expression",
       "binary_expression NE binary_expression",
       "binary_expression AND binary_expression",
       "binary_expression OR binary_expression", )
    def binary_expression(self, p):
        debug_print(
            f"binary_expression (binary_expression (TIMES | DIVIDE | MOD | PLUS | MINUS | LT | GT | GE | EQ | NE | AND | OR) binary_expression): {list(p)}")
        # TODO: Complete
        return BinaryOp(op=p[1], left=p[0], right=p[2], coord=p[0].coord)

    @_("postfix_expression")
    def unary_expression(self, p):
        debug_print(
            f"unary_expression (postfix_expression): {list(p)}")
        # TODO: Complete
        return p[0]

    @_("unary_operator unary_expression")
    def unary_expression(self, p):
        debug_print(
            f"unary_expression (unary_operator unary_expression): {list(p)}")
        # TODO: Complete
        return UnaryOp(p[0], p[1], p[1].coord)

    @_("primary_expression")
    def postfix_expression(self, p):
        debug_print(
            f"postfix_expression (primary_expression): {list(p)}")
        # TODO: Complete
        return p[0]

    @_("postfix_expression LBRACKET expression RBRACKET")
    def postfix_expression(self, p):
        debug_print(
            f"postfix_expression (postfix_expression LBRACKET expression RBRACKET): {list(p)}")
        # TODO: Complete
        return ArrayRef(name=p[0], subscript=p[2], coord=p[0].coord)

    @_("postfix_expression LPAREN optional_argument_expression RPAREN")
    def postfix_expression(self, p):
        debug_print(
            f"postfix_expression (postfix_expression LPAREN optional_argument_expression RPAREN): {list(p)}")
        # TODO: Complete
        return FuncCall(name=p[0], args=p[2], coord=p[0].coord)

    @_("argument_expression")
    def optional_argument_expression(self, p):
        debug_print(
            f"optional_argument_expression (argument_expression): {list(p)}")
        # TODO: Complete
        return p[0]

    @_("empty")
    def optional_argument_expression(self, p):
        debug_print(
            f"optional_argument_expression (empty): {list(p)}")
        pass

    @_("LPAREN expression RPAREN")
    def primary_expression(self, p):
        debug_print(
            f"primary_expression (LPAREN expression RPAREN): {list(p)}")
        return p[1]

    @_("identifier", "constant", "string")
    def primary_expression(self, p):
        debug_print(
            f"primary_expression (identifier | constant | string): {list(p)}")
        return p[0]

    @_("INT_CONST")
    def constant(self, p):
        debug_print(
            f"constant (INT_CONST): {list(p)}")
        return Constant("int", p[0], coord=self._token_coord(p))

    @_("CHAR_CONST")
    def constant(self, p):
        debug_print(
            f"constant (CHAR_CONST): {list(p)}")
        return Constant("char", p[0], coord=self._token_coord(p))

    @_("ID")
    def identifier(self, p):
        debug_print(
            f"identifier (ID): {list(p)}")
        # TODO: Complete
        return ID(p[0], coord=self._token_coord(p))

    @_("STRING_LITERAL")
    def string(self, p):
        debug_print(
            f"string (STRING_LITERAL): {list(p)}")
        # TODO: Complete
        return Constant("string", p[0], coord=self._token_coord(p))

    # Handle tokens used to identify errors
    @_("UNTERMINATED_COMMENT", "UNTERMINATED_STRING")
    def unterminated_token(self, p):
        debug_print(
            f"unterminated_token (UNTERMINATED_COMMENT | UNTERMINATED_STRING): {list(p)}")
        pass

    @_("assignment_expression", "expression COMMA assignment_expression")
    def expression(self, p):
        debug_print(
            f"expression (assignment_expression | expression COMMA assignment_expression): {list(p)}")
        if len(p) == 1:
            return p[0]
        else:
            if not isinstance(p[0], ExprList):
                p[0] = ExprList(exprs=[p[0]], coord=p[0].coord)

            p[0].exprs.append(p[2])
            return p[0]

    @_('assignment_expression')
    def argument_expression(self, p):
        debug_print(
            f"argument_expression (assignment_expression): {list(p)}")
        # TODO: Complete
        return p[0]

    @_('argument_expression COMMA assignment_expression')
    def argument_expression(self, p):
        debug_print(
            f"argument_expression (argument_expression COMMA assignment_expression): {list(p)}")
        # TODO: Complete
        arg_expr = None
        if type(p[0]) != ExprList:
            arg_expr = ExprList(exprs=[p[0], p[2]], coord=p[0].coord)
        else:
            arg_expr = p[0]
            arg_expr.exprs.append(p[2])
        return arg_expr

    @_('binary_expression')
    def assignment_expression(self, p):
        debug_print(
            f"assignment_expression (binary_expression): {list(p)}")

        return p[0]

    @_('unary_expression EQUALS assignment_expression')
    def assignment_expression(self, p):
        debug_print(
            f"assignment_expression (unary_expression EQUALS assignment_expression): {list(p)}")

        return Assignment(lvalue=p[0], op=p[1], rvalue=p[2], coord=p[0].coord)

    @_('PLUS', 'MINUS', 'NOT')
    def unary_operator(self, p):
        debug_print(
            f"unary_operator (PLUS | MINUS | NOT): {list(p)}")

        return p[0]

    @_('parameter_declaration')
    def parameter_list(self, p):
        debug_print(
            f"parameter_list (parameter_declaration): {list(p)}")

        return ParamList(params=[p[0]])

    @_('parameter_list COMMA parameter_declaration')
    def parameter_list(self, p):
        debug_print(
            f"parameter_list (parameter_list COMMA parameter_declaration): {list(p)}")

        return ParamList(params=p[0].params + [p[2]])

    @_('type_specifier declarator')
    def parameter_declaration(self, p):
        debug_print(
            f"parameter_declaration (type_specifier declarator): {list(p)}")
        if type(p[1]) == ArrayDecl:
            aux = p[1]
            while aux.type != None:
                aux = aux.type
            aux.type = p[0]
            return Decl(type=p[1], name=aux.declname, init=None)
        p[1].type = p[0]
        return Decl(type=p[1], name=p[1].declname, init=None)

    @_('type_specifier optional_init_declarator_list SEMI')
    def declaration(self, p):
        debug_print(
            f"declaration (type_specifier optional_init_declarator_list SEMI): {list(p)}")

        type_n: DeclType = p[0]
        if p[1] != None:
            new_list = []
            for x in p[1].decls:
                
                if type(x.type) == FuncDecl:
                    x.type.type = VarDecl(declname=None, type=type_n) 
                elif type(x.type) == ArrayDecl:
                    aux = x.type
                    while aux.type != None:
                        aux = aux.type
                    aux.type = type_n    
                else:
                    x.type = VarDecl(declname=None, type=type_n)
                new_list.append(x)

            return DeclList(decls=new_list)
        return DeclList(decls=[])

    @_('init_declarator_list')
    def optional_init_declarator_list(self, p):
        debug_print(
            f"optional_init_declarator_list (init_declarator_list): {list(p)}")

        return p[0]

    @_('empty')
    def optional_init_declarator_list(self, p):
        debug_print(
            f"optional_init_declarator_list (empty): {list(p)}")

        return None

    @_('init_declarator')
    def init_declarator_list(self, p):
        debug_print(
            f"init_declarator_list (init_declarator): {list(p)}")

        return DeclList(decls=[p[0]])

    @_('init_declarator_list COMMA init_declarator')
    def init_declarator_list(self, p):
        debug_print(
            f"init_declarator_list (init_declarator_list COMMA init_declarator): {list(p)}")

        return DeclList(decls=p[0].decls + [p[2]])

    @_('declarator')
    def init_declarator(self, p):
        debug_print(
            f"init_declarator (declarator): {list(p)}")
        
        aux = p[0]
        while type(aux) != VarDecl:
           aux = aux.type
        
        name = aux.declname 
        
        return Decl(
            name=name,
            type=p[0],
            init=None,
        )

    @_('declarator EQUALS initializer')
    def init_declarator(self, p):
        debug_print(
            f"init_declarator (declarator EQUALS initializer): {list(p)}")

        aux = p[0]
        while type(aux) != VarDecl:
           aux = aux.type
        
        name = aux.declname 
        
        return Decl(name=name, type=p[0], init=p[2])

    @_('assignment_expression')
    def initializer(self, p):
        debug_print(
            f"initializer (assignment_expression): {list(p)}")
        # TODO: Complete
        return p[0]

    @_('LBRACE optional_initializer_list RBRACE')
    def initializer(self, p):
        debug_print(
            f"initializer (LBRACE optional_initializer_list RBRACE): {list(p)}")
        # TODO: Complete
        return p[1]

    @_('LBRACE initializer_list COMMA RBRACE')
    def initializer(self, p):
        debug_print(
            f"initializer (LBRACE initializer_list COMMA RBRACE): {list(p)}")
        # TODO: Complete
        return p

    @_('initializer_list')
    def optional_initializer_list(self, p):
        debug_print(
            f"optional_initializer_list (initializer_list): {list(p)}")
        # TODO: Complete
        return p[0]

    @_('empty')
    def optional_initializer_list(self, p):
        debug_print(
            f"optional_initializer_list (empty): {list(p)}")
        pass

    @_('initializer')
    def initializer_list(self, p):
        debug_print(
            f"initializer_list (initializer): {list(p)}")
        # TODO: Complete
        return InitList(exprs=[p[0]], coord=p[0].coord)

    @_('initializer_list COMMA initializer')
    def initializer_list(self, p):
        debug_print(
            f"initializer_list (initializer_list COMMA initializer): {list(p)}")
        # TODO: Complete
        p[0].exprs.append(p[2])
        return p[0]

    @_('LBRACE zero_or_more_declarations zero_or_more_statements RBRACE')
    def compound_statement(self, p):
        debug_print(
            f"compound_statement (LBRACE zero_or_more_declarations zero_or_more_statements RBRACE): {list(p)}")
        
        staments = []
        for decl_list in p[1]:
            staments += decl_list.decls
        staments += p[2]
        return Compound(staments= staments, coord=self._token_coord(p))

    @_('declaration zero_or_more_declarations')
    def zero_or_more_declarations(self, p):
        debug_print(
            f"zero_or_more_declarations (declaration zero_or_more_declarations): {list(p)}")
        return [p[0]] + p[1]

    @_('empty')
    def zero_or_more_declarations(self, p):
        debug_print(
            f"zero_or_more_declarations (empty): {list(p)}")
        return []

    @_('statement zero_or_more_statements')
    def zero_or_more_statements(self, p):
        debug_print(
            f"zero_or_more_statements (statement zero_or_more_statements): {list(p)}")

        return [p[0]] + p[1]

    @_('empty')
    def zero_or_more_statements(self, p):
        debug_print(
            f"zero_or_more_statements (empty): {list(p)}")

        return []

    @_('expression_statement',
       'compound_statement',
       'selection_statement',
       'iteration_statement',
       'jump_statement',
       'assert_statement',
       'print_statement',
       'read_statement')
    def statement(self, p):
        debug_print(
            f"statement (expression_statement | compound_statement | selection_statement | iteration_statement | jump_statement | assert_statement | print_statement | read_statement): {list(p)}")

        return p[0]

    @_('optional_expression SEMI')
    def expression_statement(self, p):
        debug_print(
            f"expression_statement (optional_expression SEMI): {list(p)}")

        return p[0]

    @_('expression')
    def optional_expression(self, p):
        debug_print(
            f"optional_expression (expression): {list(p)}")

        return p[0]

    @_('empty')
    def optional_expression(self, p):
        debug_print(
            f"optional_expression (empty): {list(p)}")

        return None

    @_('IF LPAREN expression RPAREN statement',)
    def selection_statement(self, p):
        debug_print(
            f"selection_statement (IF LPAREN expression RPAREN statement): {list(p)}")

        return If(cond=p[2], iftrue=p[4], iffalse=None, coord=self._token_coord(p))

    @_('IF LPAREN expression RPAREN statement ELSE statement',)
    def selection_statement(self, p):
        debug_print(
            f"selection_statement (IF LPAREN expression RPAREN statement ELSE statement): {list(p)}")

        return If(cond=p[2], iftrue=p[4], iffalse=p[6], coord=self._token_coord(p))

    @_('FOR LPAREN optional_expression SEMI optional_expression SEMI optional_expression RPAREN statement')
    def iteration_statement(self, p):
        debug_print(
            f"iteration_statement (FOR LPAREN optional_expression SEMI optional_expression SEMI optional_expression RPAREN statement): {list(p)}")

        return For(init=p[2], cond=p[4], next=p[6], body=p[8], coord=self._token_coord(p))

    @_('FOR LPAREN declaration optional_expression SEMI optional_expression RPAREN statement')
    def iteration_statement(self, p):
        debug_print(
            f"iteration_statement (FOR LPAREN declaration optional_expression SEMI optional_expression RPAREN statement): {list(p)}")

        p[2].coord = self._token_coord(p)
        return For(init=p[2], cond=p[3], next=p[5], body=p[7], coord=self._token_coord(p))

    @_('WHILE LPAREN expression RPAREN statement')
    def iteration_statement(self, p):
        debug_print(
            f"iteration_statement (WHILE LPAREN expression RPAREN statement): {list(p)}")

        return While(cond=p[2], body=p[4], coord=self._token_coord(p))

    @_('BREAK SEMI')
    def jump_statement(self, p):
        debug_print(
            f"jump_statement (BREAK SEMI): {list(p)}")

        return Break(coord=self._token_coord(p))

    @_('RETURN optional_expression SEMI')
    def jump_statement(self, p):
        debug_print(
            f"jump_statement (RETURN optional_expression SEMI): {list(p)}")

        return Return(expr=p[1], coord=self._token_coord(p))

    @_('ASSERT expression SEMI')
    def assert_statement(self, p):
        debug_print(
            f"assert_statement (ASSERT expression SEMI): {list(p)}")

        return Assert(expr=p[1], coord=self._token_coord(p))

    @_('PRINT LPAREN optional_expression RPAREN SEMI')
    def print_statement(self, p):
        debug_print(
            f"print_statement (PRINT LPAREN optional_expression RPAREN SEMI): {list(p)}")

        return Print(expr=p[2], coord=self._token_coord(p))

    @_('READ LPAREN argument_expression RPAREN SEMI')
    def read_statement(self, p):
        debug_print(
            f"read_statement (READ LPAREN argument_expression RPAREN SEMI): {list(p)}")
        # TODO: Complete
        return Read(names=[p[2]], coord=self._token_coord(p))


def main():
    global ENABLE_STDOUT_DEBUG
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", help="Path to file to be parsed", type=str)
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    ENABLE_STDOUT_DEBUG = args.debug

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("ERROR: Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    parser = UCParser()

    # open file and print ast
    with open(input_path) as f:
        ast = parser.parse(f.read())
        print(parser.log.text)
        ast.show(buf=sys.stdout, showcoord=True)


if __name__ == "__main__":
    main()
