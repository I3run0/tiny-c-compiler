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

    # Solve ambiguity
    precedence = ()

    @_("global_declaration_list", "unterminated_token")
    def program(self, p):
        return Program(p[0])

    # Handle tokens used to identify errors
    @_("UNTERMINATED_COMMENT", "UNTERMINATED_STRING")
    def unterminated_token(self, p):
        pass

    @_("global_declaration", "global_declaration_list global_declaration")
    def global_declaration_list(self, p):
        return [p[0]] if len(p) == 1 else p[0] + [p[1]]

    @_("declaration")
    def global_declaration(self, p):
        return GlobalDecl(p[0])

    @_("function_definition")
    def global_declaration(self, p):
        return p[0]

    @_("assignment_expression", "expression COMMA assignment_expression")
    def expression(self, p):
        if len(p) == 1:
            return p[0]
        else:
            if not isinstance(p[0], ExprList):
                p[0] = ExprList(exprs=[p[0]], coord=p[0].coord)

            p[0].exprs.append(p[2])
            return p[0]

    def error(self, p):
        if p:
            self._parser_error(
                "Before %s" % p.value, Coord(p.lineno, self.uclex.find_tok_column(p))
            )
        else:
            self._parser_error("At the end of input (%s)" % self.uclex.filename)


def main():
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to file to be parsed", type=str)
    args = parser.parse_args()

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
