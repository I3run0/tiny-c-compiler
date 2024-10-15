import argparse
import pathlib
import sys
from typing import Dict, List, Tuple

from uc.uc_ast import (
    Compound,
    Decl,
    ExprList,
    FuncDef,
    Node,
    ParamList,
    Print,
    VarDecl,
    Assert,
    GlobalDecl,
    FuncDecl,
    If,
    For,
    While,
    Break,
    Read,
    Return,
    ID,
    BinaryOp,
    UnaryOp,
    InitList
)
from uc.uc_block import (
    CFG,
    BasicBlock,
    Block,
    ConditionBlock,
    EmitBlocks,
    format_instruction,
)
from uc.uc_interpreter import Interpreter
from uc.uc_parser import UCParser
from uc.uc_sema import NodeVisitor, Visitor

binary_ops = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "%": "mod",
}


class CodeGenerator(NodeVisitor):
    """
    Node visitor class that creates 3-address encoded instruction sequences
    with Basic Blocks & Control Flow Graph.
    """

    def __init__(self, viewcfg: bool, debug_print: bool):
        # To debug with debug_print
        self._enable_stdout_debug = debug_print
        self.viewcfg: bool = viewcfg
        self.current_block: Block = None

        # version dictionary for temporaries. We use the name as a Key
        self.fname: str = "_glob_"
        self.versions: Dict[str, int] = {self.fname: 0}
        # version dictionary for labels to avoid collisions
        self.label_versions: Dict[str, int] = {"if": 0, "while": 0, "for": 0}

        # The generated code (list of tuples)
        # At the end of visit_program, we call each function definition to emit
        # the instructions inside basic blocks. The global instructions that
        # are stored in self.text are appended at beginning of the code
        self.code: List[Tuple[str]] = []

        self.text: List[Tuple[str]] = (
            []
        )  # Used for global declarations & constants (list, strings)

        # TODO: Complete if needed.

    def debug_print(self, msg):
        if self._enable_stdout_debug:
            print(msg, file=sys.stdout)

    def show(self, buf=sys.stdout):
        _str = ""
        for _code in self.code:
            _str += format_instruction(_code) + "\n"
        buf.write(_str)

    def new_temp(self) -> str:
        """
        Create a new temporary variable of a given scope (function name).
        """
        if self.fname not in self.versions:
            self.versions[self.fname] = 1
        name = "%" + "%d" % (self.versions[self.fname])
        self.versions[self.fname] += 1
        return name

    def new_temp_label(self, type: str) -> str:
        """
        Create a new label an if.then, if.end, if.else, for.cond, etc.
        """
        if type not in self.label_versions:
            self.label_versions[type] = 1
        name = f"{type}" + ".%d" % (self.label_versions[type])
        self.label_versions[type] += 1
        return name

    def new_text(self, typename: str) -> str:
        """
        Create a new literal constant on global section (text).
        """
        name = "@." + typename + "." + "%d" % (self.versions["_glob_"])
        self.versions["_glob_"] += 1
        return name

    # You must implement visit_Nodename methods for all of the other
    # AST nodes.  In your code, you will need to make instructions
    # and append them to the current block code list.
    #
    # A few sample methods follow. Do not hesitate to complete or change
    # them if needed.

    def visit_Constant(self, node: Node):
        if hasattr(node.type, 'name') and node.type.name == "string":
            _target = self.new_text("str")
            inst = ("global_string", _target, node.value)
            self.text.append(inst)
        else:
            # Create a new temporary variable name
            _target = self.new_temp()
            # Make the SSA opcode and append to list of generated instructions
            inst = ("literal_" + node.type, node.value, _target)
            self.current_block.append(inst)
        # Save the name of the temporary variable where the value was placed
        node.gen_location = _target

    def visit_BinaryOp(self, node: Node):
        # Visit the left and right expressions
        self.visit(node.left)
        self.visit(node.right)

        # TODO:
        # - Load the location containing the left expression
        # - Load the location containing the right expression

        # Make a new temporary for storing the result
        target = self.new_temp()

        # Create the opcode and append to list
        opcode = binary_ops[node.op] + "_" + node.left.type.name
        inst = (opcode, node.left.gen_location,
                node.right.gen_location, target)
        self.current_block.append(inst)

        # Store location of the result on the node
        node.gen_location = target

    def visit_Print(self, node: Node):
        # Visit the expression
        self.visit(node.expr)

        # TODO: Load the location containing the expression

        # Create the opcode and append to list
        inst = ("print_" + node.expr.uc_type.typename, node.expr.gen_location)
        self.current_block.append(inst)

        # TODO: Handle the cases when node.expr is None or ExprList

    def visit_VarDecl(self, node: Node):
        """
        Allocate the variable (global or local) with the correct initial value (if there is any).
        """
        # Allocate on stack memory

        _varname = "%" + node.declname.name
        inst = ("alloc_" + node.type.name, _varname)

        self.current_block.append(inst)

        # Store optional init val
        _init = node.declname.init if hasattr(node.declname, 'init') else None
        if _init is not None:
            self.visit(_init)
            inst = (
                "store_" + node.type.name,
                _init.gen_location,
                node.declname.gen_location,
            )
            self.current_block.append(inst)

    def visit_Program(self, node: Node):
        """
        Start by visiting all global declarations. Then, visit all the function definitions and emit the code stored inside basic blocks.
        """
        # Visit all of the global declarations
        for _decl in node.gdecls:
            self.visit(_decl)
        # At the end of codegen, first init the self.code with
        # the list of global instructions allocated in self.text
        self.code = self.text.copy()
        # Also, copy the global instructions into the Program node
        node.text = self.text.copy()
        # After, visit all the function definitions and emit the
        # code stored inside basic blocks.

        self.debug_print(self.current_block)
        for _decl in node.gdecls:
            if isinstance(_decl, FuncDef):
                # _decl.cfg contains the Control Flow Graph for the function
                # cfg points to start basic block
                bb = EmitBlocks()
                bb.visit(_decl.cfg)
                for _code in bb.code:
                    self.code.append(_code)

        if self.viewcfg:  # evaluate to True if -cfg flag is present in command line
            for _decl in node.gdecls:
                if isinstance(_decl, FuncDef):
                    dot = CFG(_decl.decl.name.name)
                    # _decl.cfg contains the CFG for the function
                    dot.view(_decl.cfg)

    def visit_FuncDef(self, node: FuncDef):
        """
        Initialize the necessary blocks to construct the CFG of the function. Visit the function declaration. Visit all the declarations within the function. After allocating all declarations, visit the arguments initialization. Visit the body of the function to generate its code. Finally, setup the return block correctly and generate the return statement (even for void function).
        """
        # # Create the function block
        # # Use the function name as label
        # node.cfg = BasicBlock(node.decl.name.name)
        # self.current_block = node.cfg
        # self.current_temp = 0  # Set the temporary instruction index

        # Visit the function declaration
        self.visit(node.decl)

        # Visit the function body
        self.visit(node.body)

        # Setup the return
        func_exit = ('exit:',)
        self.current_block.append(func_exit)

    def visit_ParamList(self, node: ParamList):
        """
        Just visit all arguments.
        """
        for param in node.params:
            self.visit(param)

    def visit_GlobalDecl(self, node: GlobalDecl):
        """
        Visit each global declaration that are not function declarations. Indeed, it is usually simpler to visit the function declaration when visiting the definition of the function to generate all code at once.
        """
        for _decl in node.decls:
            if not isinstance(_decl, FuncDecl):
                self.visit(_decl)

    def visit_Decl(self, node: Decl):
        """
        Visit the type of the node (i.e., VarDecl, FuncDecl, etc.).
        """
        self.visit(node.type)

    def visit_ArrayDecl(self, node: Node):
        """
        Visit the node type.
        """
        pass

    def visit_FuncDecl(self, node: Node):
        """
        Generate the function definition (including function name, return type and arguments types). This is also a good time to generate the entry point for function, allocate a temporary for the return statement (if not a void function), and visit the arguments.
        """
        # Generate Function Definition
        _func_sig: VarDecl = node.type
        _func_param_types = node.uc_type.parameters_type
        func_definition = (
            f'define_{_func_sig.type.name}',
            f'@{_func_sig.declname.name}',
            [(_func_param_types[i], f"%{self.new_temp()}")
             for i in range(len(_func_param_types))]
        )
        self.current_block.append(func_definition)

        # Generate the Entry point
        entry = ('entry:',)
        self.current_block.append(entry)

        # Generate the temp retun
        if _func_sig.type.name != 'void':
            temp_return = (
                f'alloc_f{_func_sig.type.name}', f'%{self.new_temp()}')
            self.current_block.append(temp_return)

        # Visit function arguments
        if node.params != None:
            func_parmeters: ParamList = node.params
            for param in func_parmeters.params:
                self.visit(param)

    def visit_DeclList(self, node: Node):
        """
        Visit all of the declarations that appear inside for statement.
        """
        pass

    def visit_Type(self, node: Node):
        """
        Do nothing: just pass.
        """
        pass

    def visit_If(self, node: If):
        """
        First, generate the evaluation of the condition (visit it). Create the required blocks and the branch for the condition. Move to the first block and generate the statement related to the then, create the branch to exit. In case, there is an else block, generate it in a similar way.
        """
        self.visit(node.cond)

        then_label = self.new_temp_label("if.then")
        end_label = self.new_temp_label("if.end")

        inst = ("cbranch", node.cond.gen_location, then_label, end_label)
        self.current_block.append(inst)

        # then_block = BasicBlock(self.new_temp_label(then_label))
        # # self.visit(node.iftrue)

        # then_block.

        # self.current_block.append(ConditionBlock(then_label))
        # self.current_block.append(ConditionBlock(end_label))

    def visit_For(self, node: Node):
        """
        First, generate the initialization of the For and creates all the blocks required. Then, generate the jump to the condition block and generate the condition and the correct conditional branch. Generate the body of the For followed by the jump to the increment block. Generate the increment and the correct jump.
        """
        pass

    def visit_While(self, node: Node):
        """
        The generation of While is similar to For except that it does not require the part related to initialization and increment.
        """
        pass

    def visit_Compound(self, node: Compound):
        """
        Visit the list of block items (declarations or statements).
        """
        for statment in node.staments:
            self.visit(statment)

    def visit_Assignment(self, node: Node):
        """
        First, visit right side and load the value according to its type. Then, visit the left side and generate the code according to the assignment operator and the type of the expression (ID or ArrayRef).
        """
        pass

    def visit_Break(self, node: Node):
        """
        Generate a jump instruction to the current exit label.
        """
        pass

    def visit_FuncCall(self, node: Node):
        """
        Start by generating the code for the arguments: for each one of them, visit the expression and generate a param_type instruction with its value. Then, allocate a temporary for the return value and generate the code to call the function.
        """
        pass

    def visit_Assert(self, node: Assert):
        """
        The assert is a conditional statement which generate code quite similar to the If Block. If the expression is false, the program should issue an error message (assertfail) and terminate. If the expression is true, the program proceeds to the next sentence.

        Visit the assert condition. Create the blocks for the condition and adust their predecessors. Generate the branch instruction and adjust the blocks to jump according to the condition. Generate the code for unsuccessful assert, generate the print instruction and the jump instruction to the return block, and successful assert.
        """
        self.visit(node.expr)

        pass

    def visit_EmptyStatement(self, node: Node):
        pass

    def visit_Print(self, node: Print):
        if node.expr != None:
            # TODO: Maybe handle ExprList differently
            # if isinstance(node, ExprList):
            #     pass  # Todo
            self.visit(node.expr)  # Single expression
        else:
            print_void = ('print_void',)
            self.current_block.append(print_void)

    def visit_Read(self, node: Node):
        pass

    def visit_Return(self, node: Node):
        pass

    '''
    def visit_Constant(self, node: Node):
        pass
    '''

    def visit_ID(self, node: Node):
        if isinstance(node.parent, Decl):
            pass
        else:
            node.gen_location = self.new_temp()
            self.current_block.append(('load', node.name, node.gen_location))
            self.debug_print(f"Parent: {node.parent}")

    def visit_BinaryOp(self, node: Node):
        pass

    def visit_UnaryOp(self, node: Node):
        pass

    def visit_ExprList(self, node: Node):
        pass

    def visit_InitList(self, node: Node):
        pass


def main():
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to file to be used to generate uCIR. By default, this script only runs the interpreter on the uCIR. \
              Use the other options for printing the uCIR, generating the CFG or for the debug mode.",
        type=str,
    )
    parser.add_argument(
        "--ir",
        help="Print uCIR generated from input_file.",
        action="store_true",
    )
    parser.add_argument(
        "--cfg", help="Show the cfg of the input_file.", action="store_true"
    )
    parser.add_argument(
        "--debug", help="Run interpreter in debug mode.", action="store_true"
    )
    parser.add_argument(
        "--debug_print", action="store_true"
    )

    args = parser.parse_args()

    print_ir = args.ir
    create_cfg = args.cfg
    interpreter_debug = args.debug
    enable_stdout_debug_print = args.debug_print

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

    gen = CodeGenerator(create_cfg, enable_stdout_debug_print)
    gen.visit(ast)
    gencode = gen.code

    if print_ir:
        print("Generated uCIR: --------")
        gen.show()
        print("------------------------\n")

    vm = Interpreter(interpreter_debug)
    vm.run(gencode)


if __name__ == "__main__":
    main()
