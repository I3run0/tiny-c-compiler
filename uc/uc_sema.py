import argparse
import pathlib
import sys
from copy import deepcopy
from typing import Any, Dict, Union

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
    Node,
)
from uc.uc_parser import UCParser
from uc.uc_type import (
    ArrayType,
    BoolType,
    CharType,
    FunctionType,
    IntType,
    StringType,
    VoidType,
    uCType,
    BasicVariableTypes,
)

ENABLE_STDOUT_DEBUG = False


def node_to_debug_string(node):
    if ENABLE_STDOUT_DEBUG:
        clone = deepcopy(node)
        if hasattr(clone, "parent"):
            delattr(clone, "parent")

        for attr, value in clone.__dict__.items():
            if issubclass(type(value), Node):
                clone.__dict__[attr] = node_to_debug_string(value)

            if isinstance(value, list):
                for i, item in enumerate(value):
                    if issubclass(type(item), Node):
                        clone.__dict__[attr][i] = node_to_debug_string(item)

        return clone


class SymbolTable:
    """Class representing a symbol table.

    `add` and `lookup` methods are given, however you still need to find a way to
    deal with scopes.

    ## Attributes
    - :attr data: the content of the SymbolTable
    """

    def __init__(self) -> None:
        """ Initializes the SymbolTable. """
        self.__data_stack = [dict()]

    @property
    def current_scope(self) -> Dict[str, Any]:
        """ Returns a copy of the SymbolTable current scope.
        """
        return deepcopy(self.__data_stack[-1])

    def add(self, name: str, type: uCType, scope=-1) -> None:
        """ Adds (or overwrites) to the SymbolTable.

        ## Parameters
        - :param name: the identifier on the SymbolTable
        - :param value: the value to assign to the given `name`
        """
        self.__data_stack[scope][name] = SymbolTableEntry(self, type)

    def get_last_scope(self, name: str) -> None:
        if name in self.__data_stack[-1]:
            return self.__data_stack[-1][name]

    def lookup(self, name: str) -> Union["SymbolTableEntry", None]:
        """ Searches `name` on the SymbolTable and returns the value
        assigned to it.

        ## Parameters
        - :param name: the identifier that will be searched on the SymbolTable

        ## Return
        - :return: the value assigned to `name` on the SymbolTable. If `name` is not found, `None` is returned.
        """
        for data in reversed(self.__data_stack):
            if name in data:
                return data.get(name)

    def push_scope(self) -> None:
        """ Creates a new scope on the SymbolTable. """
        self.__data_stack.append(dict())
        self.__data = self.__data_stack[-1]

    def pop_scope(self) -> None:
        """ Removes the last scope from the SymbolTable. """
        self.__data_stack.pop()
        self.__data = self.__data_stack[-1]

    def __str__(self) -> None:
        return str(self.__data_stack)


class SymbolTableEntry:
    def __init__(self, scope: SymbolTable, type: uCType) -> None:
        """ Initializes the SymbolTableEntry. """
        self.scope = scope
        self.type = type


class NodeVisitor:
    """A base NodeVisitor class for visiting uc_ast nodes.
    Subclass it and define your own visit_XXX methods, where
    XXX is the class name you want to visit with these
    methods.
    """
    _enable_stdout_debug = ENABLE_STDOUT_DEBUG
    _method_cache = None

    def visit(self, node):
        """Visit a node."""

        if self._method_cache is None:
            self._method_cache = {}

        visitor = self._method_cache.get(node.__class__.__name__)
        if visitor is None:
            method = "visit_" + node.__class__.__name__
            visitor = getattr(self, method, self.generic_visit)
            self._method_cache[node.__class__.__name__] = visitor

        self.debug_print_visit(node)
        self.anotate_parent_visit(node)
        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a
        node. Implements preorder visiting of the node.
        """
        for _, child in node.children():
            self.visit(child)

    def anotate_parent_visit(self, node):
        for _, child in node.children():
            # debug_print(f"Annotating {type(child).__name__} with parent {
            #             type(node).__name__}")
            child.parent = node

    def debug_print_visit(self, node):
        if self._enable_stdout_debug:
            clone = deepcopy(node)
            if hasattr(clone, "parent"):
                delattr(clone, "parent")
            print(f"Visiting {clone}", file=sys.stdout)


class Visitor(NodeVisitor):
    """
    Program visitor class. This class uses the visitor pattern. You need to define methods
    of the form visit_NodeName() for each kind of AST node that you want to process.
    """

    def __init__(self):
        # Initialize the symbol table
        self.symtab = SymbolTable()
        self.typemap = {
            "int": IntType,
            "char": CharType,
            "bool": BoolType,
            "void": VoidType,
            "string": StringType,
        }
        self._enable_stdout_debug = ENABLE_STDOUT_DEBUG
        # TODO: Complete...

    def _assert_semantic(self, condition: bool, msg_code: int, coord, name: str = "", ltype="", rtype=""):
        """Check condition, if false print selected error message and exit"""
        error_msgs = {
            1: f"{name} is not defined",
            2: f"subscript must be of type(int), not {ltype}",
            3: "Expression must be of type(bool)",
            4: f"Cannot assign {rtype} to {ltype}",
            5: f"Binary operator {name} does not have matching LHS/RHS types",
            6: f"Binary operator {name} is not supported by {ltype}",
            7: "Break statement must be inside a loop",
            8: "Array dimension mismatch",
            9: f"Size mismatch on {name} initialization",
            10: f"{name} initialization type mismatch",
            11: f"{name} initialization must be a single element",
            12: "Lists have different sizes",
            13: "List & variable have different sizes",
            14: f"conditional expression is {ltype}, not type(bool)",
            15: f"{name} is not a function",
            16: f"no. arguments to call {name} function mismatch",
            17: f"Type mismatch with parameter {name}",
            18: "The condition expression must be of type(bool)",
            19: "Expression must be a constant",
            20: "Expression is not of basic type",
            21: f"{name} does not reference a variable of basic type",
            22: f"{name} is not a variable",
            23: f"Return of {ltype} is incompatible with {rtype} function definition",
            24: f"Name {name} is already defined in this scope",
            25: f"Unary operator {name} is not supported",
        }
        if not condition:
            msg = error_msgs[msg_code]  # invalid msg_code raises Exception
            print("SemanticError: %s %s" % (msg, coord), file=sys.stdout)
            sys.exit(1)

    def visit_Program(self, node):
        # Visit all of the global declarations
        for _decl in node.gdecls:
            self.visit(_decl)

    def _visit_FuncDef_assert_returns(self, compound_node, func_type):
        if isinstance(compound_node, Return):
            _ret_type = compound_node.expr.uc_type if hasattr(compound_node.expr,  'uc_type')\
                else self.typemap['void']
            self._assert_semantic(
                _ret_type == func_type, 23, coord=compound_node.coord,
                ltype=_ret_type, rtype=func_type)

        if isinstance(compound_node, Compound) and compound_node.staments != []:
            for statement in compound_node.staments:
                if isinstance(statement, If):
                    self._visit_FuncDef_assert_returns(
                        statement.iftrue, func_type)
                    if statement.iffalse is not None:
                        self._visit_FuncDef_assert_returns(
                            statement.iffalse, func_type)

                elif isinstance(statement, While) or\
                        isinstance(statement, For):
                    self._visit_FuncDef_assert_returns(
                        statement.body, func_type)

                else:
                    self._visit_FuncDef_assert_returns(statement, func_type)

    def visit_FuncDef(self, node):
        '''
        Initialize the list of declarations that appears inside loops.
        Save the reference to current function. Visit the return type of
        the Function, the function declaration, the parameters, and the function body.
        '''
        self.symtab.push_scope()

        self.visit(node.type)

        self.visit(node.decl)

        self.symtab.add(node.decl.name.name, node.decl.type.uc_type, -2)
        self.visit(node.body)

        # check the function return type
        _func_type = node.type.uc_type
        if len(node.body.staments) == 0:
            self._assert_semantic(
                self.typemap['void'] == _func_type, 23, node.body.coord,
                ltype=self.typemap['void'], rtype=_func_type
            )
        else:
            self._visit_FuncDef_assert_returns(node.body, node.type.uc_type)

        self.symtab.pop_scope()

    def visit_ParamList(self, node):
        for param in node.params:
            self.visit(param)

    def visit_GlobalDecl(self, node):
        for _decl in node.decls:
            self.visit(_decl)

    def visit_Decl(self, node):
        # self.visit(node.name)

        node.type.declname = node.name

        self.visit(node.type)
        if node.init:
            self.visit(node.init)

        # self.symtab.add(node.name.name, node.type.uc_type)
        # print(str(self.symtab))
        if node.init:
            if type(node.type.uc_type) == ArrayType:
                def assert_init_list_has_same_size(id, init_nodes):
                    next_level_nodes = []

                    if any([type(init_node) != InitList for init_node in init_nodes]):
                        return

                    level_list_size = len(init_nodes[0].exprs)
                    for init_node in init_nodes:
                        self._assert_semantic(
                            len(init_node.exprs) == level_list_size, 12, coord=id.coord)
                        next_level_nodes += init_node.exprs

                    if len(next_level_nodes) > 0:
                        assert_init_list_has_same_size(id, next_level_nodes)

                assert_init_list_has_same_size(node.name, [node.init])

                # Implement logic to traverse init and decl together
                # if decl node is an array, init node must be init list
                # if decl node is primitive type, init node must be a constant
                # Handle nested arrays
                def assert_init_list_size_matches_array_dimension(id, array_node, init_node):
                    if init_node is not None:
                        self._assert_semantic(
                            type(init_node) == InitList or
                            (array_node.uc_type.type == CharType and init_node.uc_type ==
                             StringType), 4, id.coord,
                            ltype=array_node.type.uc_type, rtype=init_node.uc_type,
                            name=id.name)

                        if hasattr(array_node.uc_type, "size"):
                            if type(init_node) == InitList:
                                self._assert_semantic(
                                    array_node.uc_type.size is None or
                                    len(init_node.exprs) == array_node.uc_type.size, 13,
                                    node.name.coord)
                            else:  # node.init is Constant(type=string)
                                self._assert_semantic(
                                    (array_node.uc_type.size is None) or
                                    (len(init_node.value) ==
                                     array_node.uc_type.size),
                                    9, name=node.name.name, coord=node.name.coord)

                        if not hasattr(array_node.uc_type, "size") or \
                           array_node.uc_type.size is None:
                            if type(init_node) == InitList:
                                array_node.uc_type.size = len(init_node.exprs)
                            else:
                                array_node.uc_type.size = len(init_node.value)

                        if type(init_node) == InitList:
                            for _expr in init_node.exprs:
                                if not isinstance(array_node.type, ArrayDecl):
                                    self._assert_semantic(
                                        _expr.uc_type == array_node.type.uc_type, 10,
                                        id.coord, name=id.name)

                                    # If the current array element type isn't an array,
                                    # all initialization elements must be constants
                                    self._assert_semantic(
                                        isinstance(_expr, Constant), 19,
                                        name=id.name, coord=_expr.coord)
                                else:
                                    assert_init_list_size_matches_array_dimension(
                                        id, array_node.type, _expr)

                assert_init_list_size_matches_array_dimension(
                    node.name, node.type, node.init)
            else:
                if type(node.init) == InitList and \
                        node.init.exprs[0].uc_type == node.type.uc_type:
                    self._assert_semantic(
                        len(node.init.exprs) == 1, 11,
                        node.name.coord, name=node.name.name)
                else:
                    self._assert_semantic(
                        node.type.uc_type == node.init.uc_type, 10,
                        node.name.coord, name=node.name.name)

    def visit_VarDecl(self, node):
        self.visit(node.type)

        if node.declname:
            name_in_scope = self.symtab.get_last_scope(node.declname.name)
            if name_in_scope == None:
                self.symtab.add(node.declname.name, node.type.uc_type)
                self.visit(node.declname)
            else:
                self._assert_semantic(
                    name_in_scope == None,
                    24,
                    name=node.declname.name,
                    coord=node.declname.coord
                )

        node.uc_type = node.type.uc_type

    def visit_ArrayDecl(self, node):
        self.visit(node.type)
        if node.dim:
            self.visit(node.dim)

            # If current array has a dimension all nested arrays must have a dimension
            current_node = node
            while isinstance(current_node, ArrayDecl):
                if current_node.dim is None:
                    current_parent = current_node.parent
                    while not hasattr(current_parent, "name"):
                        current_parent = current_parent.parent

                    self._assert_semantic(
                        False,
                        8, coord=current_parent.name.coord
                    )
                current_node = current_node.type

            node.uc_type = ArrayType(node.type.uc_type, int(node.dim.value))
        else:
            node.uc_type = ArrayType(node.type.uc_type)

        # Handle nested arrays
        current_node = node
        while not isinstance(current_node.type, VarDecl):
            current_node = current_node.type

        self.symtab.add(current_node.type.declname.name, node.uc_type)

    def visit_FuncDecl(self, node):
        funcDecl: FuncDecl = node
        self.visit(funcDecl.type)

        _params_type = None
        if funcDecl.params:
            self.visit(funcDecl.params)
            _params_type = [
                decl.type.uc_type for decl in funcDecl.params.params]
        else:
            _params_type = []

        funcDecl.uc_type = FunctionType(return_type=funcDecl.type.uc_type,
                                        parameters_type=_params_type)

        # add to the last scope
        self.symtab.add(funcDecl.declname.name, funcDecl.uc_type)

    def visit_DeclList(self, node):
        for _decl in node.decls:
            self.visit(_decl)

    def visit_Type(self, node):
        node.uc_type = self.typemap[node.name]

    def visit_If(self, node):
        self.symtab.push_scope()
        self.visit(node.cond)
        self.visit(node.iftrue)
        if node.iffalse:
            self.visit(node.iffalse)

        self._assert_semantic(
            hasattr(node.cond, "uc_type") and
            node.cond.uc_type == self.typemap["bool"], 18, node.cond.coord)

        self.symtab.pop_scope()

    def visit_For(self, node):
        self.symtab.push_scope()
        node: For = node
        self.visit(node.init)
        self.visit(node.cond)
        self.visit(node.next)
        self.visit(node.body)
        self.symtab.pop_scope()

    def visit_While(self, node):
        self.symtab.push_scope()
        self.visit(node.cond)

        self._assert_semantic(
            node.cond.uc_type == self.typemap['bool'],
            msg_code=14,
            coord=node.coord,
            ltype=node.cond.uc_type
        )
        self.visit(node.body)
        self.symtab.pop_scope()

    def visit_Compound(self, node):
        for _stament in node.staments:
            if isinstance(_stament, Compound):
                self.symtab.push_scope()
                self.visit(_stament)
                self.symtab.pop_scope()
            else:
                self.visit(_stament)

    def visit_Assignment(self, node):
        # visit right side
        self.visit(node.rvalue)
        rtype = node.rvalue.uc_type
        # visit left side (must be a location)
        _var = node.lvalue
        self.visit(_var)
        if isinstance(_var, ID):
            self._assert_semantic(_var.scope is not None,
                                  1, node.coord, name=_var.name)

        ltype = node.lvalue.uc_type

        # Check that assignment is allowed
        self._assert_semantic(ltype == rtype, 4, node.coord,
                              ltype=ltype, rtype=rtype)

        # Check that assign_ops is supported by the type
        self._assert_semantic(
            node.op in ltype.assign_ops, 5, node.coord, name=node.op, ltype=ltype
        )

    def visit_Break(self, node):
        current_parent = node.parent
        while not isinstance(current_parent, While) and \
                not isinstance(current_parent, For) and   \
                not isinstance(current_parent, Program):
            current_parent = current_parent.parent

        self._assert_semantic(
            isinstance(current_parent, While) or isinstance(
                current_parent, For),
            7,
            coord=node.coord
        )
        pass

    def visit_FuncCall(self, node):
        '''
        Verify that the given name is a function, or return an error 
        if it is not. Initialize the node type and name using the symbole table.
        Check that the number and the type of the arguments correspond to the
        parameters in the function definition or return an error.
        '''
        self.visit(node.name)

        _uc_type = node.name.uc_type

        self._assert_semantic(
            isinstance(_uc_type, FunctionType),
            msg_code=15,
            name=node.name.name,
            coord=node.coord

        )
        if node.args is not None:
            self.visit(node.args)

            if isinstance(node.args, ExprList):
                self._assert_semantic(
                    len(node.args.exprs) == len(_uc_type.parameters_type),
                    msg_code=16,
                    name=node.name.name,
                    coord=node.coord
                )

                for i in range(len(node.args.exprs)):
                    self._assert_semantic(
                        node.args.exprs[i].uc_type == _uc_type.parameters_type[i],
                        msg_code=17,
                        name=node.args.exprs[i].name if hasattr(node.args.exprs[i], 'name')
                        else None,
                        coord=node.args.exprs[i].coord
                    )

            elif isinstance(node.args, ID):
                self._assert_semantic(
                    1 == len(_uc_type.parameters_type),
                    msg_code=16,
                    name=node.name.name,
                    coord=node.coord
                )

                self._assert_semantic(
                    node.args.uc_type == _uc_type.parameters_type[0],
                    msg_code=17,
                    name=node.args.name,
                    coord=node.args.coord
                )

        node.uc_type = _uc_type.return_type

    def visit_Assert(self, node):
        self.visit(node.expr)
        self._assert_semantic(
            node.expr.uc_type == self.typemap['bool'],
            msg_code=3,
            coord=node.expr.coord
        )

    def visit_Print(self, node):
        def assert_expr(expr):
            if isinstance(expr, ID):
                self._assert_semantic(
                    expr.uc_type in BasicVariableTypes,
                    msg_code=21,
                    coord=expr.coord,
                    name=expr.name
                )
            else:
                self._assert_semantic(
                    expr.uc_type is not self.typemap['void'],
                    msg_code=20,
                    coord=expr.coord,
                )

        if node.expr is not None:
            self.visit(node.expr)

        if isinstance(node.expr, ExprList):
            for expr in node.expr.exprs:
                assert_expr(expr)
        elif node.expr is not None:
            assert_expr(node.expr)

    def visit_Read(self, node):
        for name in (node.names[0].exprs if isinstance(node.names[0], ExprList) else node.names):
            self.visit(name)
            arref_or_id = isinstance(name, ID) or isinstance(name, ArrayRef)

            _name = None
            if arref_or_id:
                _name = name.name
                if isinstance(name, ArrayRef):
                    while isinstance(_name, ArrayRef):
                        _name = _name.name
                    _name = _name.name  # After iterate through ArrayRef, foud a ID

            previous_def = self.symtab.lookup(_name) is not None

            self._assert_semantic(
                arref_or_id and previous_def,
                name=str(name).split("(")[0],
                msg_code=22,
                coord=name.coord
            )

    def visit_Return(self, node):
        if node.expr is not None:
            self.visit(node.expr)

    def visit_Constant(self, node):
        node.uc_type = self.typemap[node.type]

    def visit_ID(self, node):
        entry = self.symtab.lookup(node.name)
        if entry is not None:
            node.scope = entry
            node.uc_type = entry.type
        else:
            node.scope = None
            node.uc_type = None

    def visit_BinaryOp(self, node):
        # Visit the left and right expression
        binary_node: BinaryOp = node
        self.visit(binary_node.lvalue)
        self.visit(binary_node.rvalue)

        def assert_def(x): return self._assert_semantic(
            self.symtab.lookup(x.name) != None, 1, coord=x.coord, name=x.name)

        if isinstance(node.lvalue, ID):
            assert_def(node.lvalue)

        if isinstance(node.rvalue, ID):
            assert_def(node.rvalue)

        # If a function return the type of the function
        ltype = node.lvalue.uc_type
        rtype = node.rvalue.uc_type
        op = node.op

        self._assert_semantic(
            ltype == rtype, 5, node.coord, name=op, ltype=ltype, rtype=rtype)
        self._assert_semantic(
            op in ltype.binary_ops or op in ltype.rel_ops, 6, node.coord, name=op, ltype=ltype)

        if (op in ltype.binary_ops):
            node.uc_type = ltype
        elif (op in ltype.rel_ops):
            node.uc_type = self.typemap["bool"]

    def visit_UnaryOp(self, node):
        self.visit(node.expr)
        exprtype = node.expr.uc_type

        # Check that op is supported by the type
        self._assert_semantic(
            node.op in exprtype.unary_ops, 25, node.coord, name=node.op
        )

        node.uc_type = node.expr.uc_type

    def visit_ExprList(self, node):
        for _expr in node.exprs:
            self.visit(_expr)

    def visit_ArrayRef(self, node):
        self.visit(node.name)
        self.visit(node.subscript)

        self._assert_semantic(
            node.subscript.uc_type == IntType,
            2, node.subscript.coord, ltype=node.subscript.uc_type)

        # Copy the uc_type from the ID node to the last ArrayRef
        if Type(node.name != ArrayRef):
            node.uc_type = node.name.uc_type.type

        # check the identifier scope
        _var = node.subscript
        if isinstance(_var, ID):
            self._assert_semantic(_var.scope is not None,
                                  1, node.coord, name=_var.name)

        # check if the subscript is a IntType
        self._assert_semantic(
            node.subscript.uc_type == self.typemap["int"], 2, node.coord, ltype=node.subscript.uc_type)

    def visit_InitList(self, node):
        for _expr in node.exprs:
            self.visit(_expr)

        node.uc_type = ArrayType(node.exprs[0].uc_type, len(node.exprs))
        # TODO: Maybe check if all the elements are of the same type
        # self._assert_semantic(
        #     all(_expr.uc_type == node.uc_type.type for _expr in node.exprs), 9, node.coord, name=node.uc_type)


def main():
    global ENABLE_STDOUT_DEBUG

    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", help="Path to file to be semantically checked", type=str
    )
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    ENABLE_STDOUT_DEBUG = args.debug

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    # set error function
    p = UCParser()
    # open file and parse it
    with open(input_path) as f:
        ast = p.parse(f.read())
        sema = Visitor()
        sema.visit(ast)


if __name__ == "__main__":
    main()
