import argparse
import pathlib
import sys

from sly import Lexer


class UCLexer(Lexer):
    """A lexer for the uC language. After building it, set the
    input text with input(), and call token() to get new
    tokens.
    """

    def __init__(self, error_func):
        self.error_func = error_func
        self.filename = ""

    def reset_lineno(self):
        """Resets the internal line number counter of the lexer."""
        self.lexer.lineno = 1

    def _error(self, msg, token):
        location = self._make_tok_location(token)
        self.error_func(msg, location[0], location[1])
        self.index += 1

    def find_tok_column(self, token):
        """Find the column of the token in its line."""
        last_cr = self.text.rfind("\n", 0, token.index)
        return token.index - last_cr

    def _make_tok_location(self, token):
        return (self.lineno, self.find_tok_column(token))

    # Error handling
    def error(self, t):
        msg = f"Illegal character {t.value[0]!r}"
        self._error(msg, t)

    # Scanner (used only for test)
    def scan(self, data):
        output = ""
        for token in self.tokenize(data):
            token = (
                f"LexToken({token.type},{token.value!r},{token.lineno},{token.index})"
            )
            print(token)
            output += token + "\n"
        return output

    # Reserved keywords
    keywords = (
        "ASSERT",
        "BREAK",
        "CHAR",
        "ELSE",
        "FOR",
        "IF",
        "INT",
        "PRINT",
        "READ",
        "RETURN",
        "VOID",
        "WHILE",
    )

    keyword_map = {keyword.lower(): keyword for keyword in keywords}

    #
    # All the tokens recognized by the lexer
    #
    tokens = keywords + (
        # Identifiers
        "ID",
        # constants
        "INT_CONST",
        "CHAR_CONST",
        "STRING_LITERAL",
        # Operators
        "PLUS",
        "MINUS",
        "TIMES",
        "DIVIDE",
        "MOD",
        "OR",
        "AND",
        "NOT",
        "LT",
        "LE",
        "GT",
        "GE",
        "EQ",
        "NE",
        # Assignment
        "EQUALS",
        # Delimeters
        "LPAREN",
        "RPAREN",  # ( )
        "LBRACKET",
        "RBRACKET",  # [ ]
        "LBRACE",
        "RBRACE",  # { }
        "COMMA",
        "SEMI",  # , ;
        # Lexer errors
        "UNTERMINATED_COMMENT",
        "UNTERMINATED_STRING",
    )

    #
    # Rules
    #
    ignore = " \t"
    
    # Newlines
    @_(r'\n+')
    def ignore_newline(self, t):
        self.lineno += len(t.value)

    # Comments
    @_(r'(//.*)|(/\*(.|\n)*?\*/)') # Essa regra esta errada
    def ignore_comment(self, t):
        self.lineno += t.value.count("\n")

    @_(r'/\*(.|\n)*')
    def UNTERMINATED_COMMENT(self, t):
        msg = "Unterminated comment"
        self._error(msg, t)

    # Identifiers and keywords
    @_(r"[a-zA-Z_][a-zA-Z0-9_]*")
    def ID(self, t):
        t.type = self.keyword_map.get(t.value, "ID")
        return t

    # String literals
    @_(r'".*?"')
    def STRING_LITERAL(self, t):
        t.value = t.value[1:-1]
        return t

    @_(r'".*')
    def UNTERMINATED_STRING(self, t):
        msg = "Unterminated string"
        self._error(msg, t)

    # Continue the Lexer Rules
    # ...
    
    #Regular expression rules for tokens
    #constants
    INT_CONST = r'[0-9]+'
    CHAR_CONST = r"'.?'"
   
    #Operators
    PLUS = r'\+'
    MINUS = r'-'
    TIMES = r'\*'
    DIVIDE = r'/'
    MOD = r'%'
    OR = r'\|\|'
    AND = r'&&'
    LE = r'<='
    GE = r'>='
    EQ = r'=='
    NE = r'!='
    NOT = r'!'
    LT = r'<'
    GT = r'>'

    # Assignment
    EQUALS = r'='

    #Delimiters
    LPAREN = r'\('
    RPAREN = r'\)'
    LBRACKET = r'\['
    RBRACKET = r'\]'
    LBRACE = r'\{'
    RBRACE = r'\}'
    COMMA = r','
    SEMI = r';'

def main():
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to file to be scanned", type=str)
    args = parser.parse_args()

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    def print_error(msg, x, y):
        # use stdout to match with the output in the .out test files
        print(f"Lexical error: {msg} at {x}:{y}", file=sys.stdout)

    # Create the lexer and set error function
    lexer = UCLexer(print_error)

    # open file and print tokens
    with open(input_path) as f:
        lexer.scan(f.read())


if __name__ == "__main__":
    main()
