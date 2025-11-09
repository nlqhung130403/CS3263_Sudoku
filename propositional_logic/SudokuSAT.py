from pysat.solvers import Solver
from pysat.formula import CNF

class SudokuSATSolver:
    """
    Solves a Sudoku puzzle of size N x N by encoding it as a SAT problem.
    
    N must be a perfect square (e.g., 4, 9, 16, 25).
    The puzzle is represented by N*N*N variables: var(r, c, v)
    which is true if cell (r, c) has value v.
    r, c, v are all in the range [1, N].
    
    Values are represented as:
    1-9   -> '1'...'9'
    10-35 -> 'A'...'Z'
    Empty -> '.' or '0'
    """

    def __init__(self, puzzle_str: str, solver_name: str = "Glucose3"):
        """
        Initializes the solver for an N x N Sudoku puzzle.
        N is determined from the puzzle string length (e.g., 81 -> 9x9).
        """
        self.puzzle_str = puzzle_str
        self.grid_size = int(len(puzzle_str)**0.5)
        self.box_size = int(self.grid_size**0.5)
        
        if self.box_size * self.box_size != self.grid_size:
            raise ValueError(f"Invalid puzzle length: {len(puzzle_str)}. "
                             f"Length must be N*N where N is a perfect square (e.g., 16, 81, 256).")
        
        if self.grid_size > 35:
            raise ValueError(f"Grid size {self.grid_size}x{self.grid_size} is not supported "
                             "(max is 35x35, using values 1-9 and A-Z).")
                             
        self.solver_name = solver_name

        # self.vars[r][c][v] will store the integer ID for the SAT variable
        # We use 1-based indexing for r, c, v, so we need size N+1
        n = self.grid_size
        self.vars = [[[0 for _ in range(n + 1)] for _ in range(n + 1)] for _ in range(n + 1)]
        
        self.cnf = CNF()
        self.solution = None # Will store the N x N solution grid
        
        self._setup_variables()
        self._build_cnf()

    def _char_to_val(self, char: str) -> int | None:
        """
        Converts a puzzle character ('.', '1', 'A', etc.) to its integer value (1-N).
        Returns None for empty cells.
        """
        if char in ('.', '0', '_'): # Common empty chars
            return None
        
        val = 0
        if '1' <= char <= '9':
            val = int(char)
        elif 'A' <= char <= 'Z':
            val = ord(char) - ord('A') + 10
        elif 'a' <= char <= 'z':
            val = ord(char) - ord('a') + 10
        else:
            raise ValueError(f"Invalid character in puzzle string: '{char}'")
        
        if 1 <= val <= self.grid_size:
            return val
        else:
            # This means the puzzle has a clue 'F' (15) for a 9x9 grid.
            raise ValueError(f"Clue value '{char}' (={val}) is out of range for grid size {self.grid_size}")

    def _val_to_char(self, val: int) -> str:
        """
        Converts an integer value (1-N) from the solution back to a character.
        """
        if val == 0: # Should not happen in a solved grid, but good to have
            return '.'
        if 1 <= val <= 9:
            return str(val)
        if 10 <= val <= 35:
            return chr(ord('A') + val - 10)
        
        # This should be impossible if self.grid_size <= 35
        return '?' 

    def _setup_variables(self):
        """
        Maps each (r, c, v) tuple to a unique integer variable ID (1 to N*N*N).
        """
        var_id = 1
        for r in range(1, self.grid_size + 1):
            for c in range(1, self.grid_size + 1):
                for v in range(1, self.grid_size + 1):
                    self.vars[r][c][v] = var_id
                    var_id += 1

    def _add_cell_constraints(self):
        """
        Adds "exactly one" constraints for each cell.
        Each cell (r, c) must contain exactly one value v from {1..N}.
        """
        for r in range(1, self.grid_size + 1):
            for c in range(1, self.grid_size + 1):
                # 1. "At least one" value: (v1 | v2 | ... | vN)
                at_least_one = [self.vars[r][c][v] for v in range(1, self.grid_size + 1)]
                self.cnf.append(at_least_one)
                
                # 2. "At most one" value: (-v_i | -v_j) for all i != j
                # This is the standard pairwise "at-most-one" encoding.
                for v1 in range(1, self.grid_size + 1):
                    for v2 in range(v1 + 1, self.grid_size + 1):
                        self.cnf.append([-self.vars[r][c][v1], -self.vars[r][c][v2]])

    def _add_row_constraints(self):
        """
        Adds "at most one" constraints for each row.
        If each cell has exactly one value (from cell_constraints),
        we only need to ensure no two cells in a row have the *same* value.
        The "at least one" (each value 1-N must appear) is then implied.
        """
        for r in range(1, self.grid_size + 1):
            for v in range(1, self.grid_size + 1):
                # For a given row 'r' and value 'v',
                # no two different columns (c1, c2) can both be true.
                for c1 in range(1, self.grid_size + 1):
                    for c2 in range(c1 + 1, self.grid_size + 1):
                        self.cnf.append([-self.vars[r][c1][v], -self.vars[r][c2][v]])

    def _add_col_constraints(self):
        """
        Adds "at most one" constraints for each column.
        Symmetric to row constraints.
        """
        for c in range(1, self.grid_size + 1):
            for v in range(1, self.grid_size + 1):
                # For a given col 'c' and value 'v',
                # no two different rows (r1, r2) can both be true.
                for r1 in range(1, self.grid_size + 1):
                    for r2 in range(r1 + 1, self.grid_size + 1):
                        self.cnf.append([-self.vars[r1][c][v], -self.vars[r2][c][v]])

    def _add_box_constraints(self):
        """
        Adds "at most one" constraints for each (box_size x box_size) box.
        Symmetric to row/col constraints.
        """
        for v in range(1, self.grid_size + 1):
            for br in range(self.box_size): # Box row (0 to box_size-1)
                for bc in range(self.box_size): # Box col (0 to box_size-1)
                    
                    # Collect all N variables for value 'v' in this box
                    cells_in_box = []
                    for r_offset in range(self.box_size):
                        for c_offset in range(self.box_size):
                            r = br * self.box_size + 1 + r_offset
                            c = bc * self.box_size + 1 + c_offset
                            cells_in_box.append(self.vars[r][c][v])
                    
                    # Add pairwise "at most one" constraints
                    for i in range(len(cells_in_box)):
                        for j in range(i + 1, len(cells_in_box)):
                            self.cnf.append([-cells_in_box[i], -cells_in_box[j]])

    def _add_clue_constraints(self):
        """
        Adds unit clauses for the pre-filled cells (clues) from the puzzle string.
        """
        for i in range(len(self.puzzle_str)):
            char = self.puzzle_str[i]
            v = self._char_to_val(char)
            
            if v is not None:
                # Map 1D index (0 to N*N-1) to 2D (row, col) (1 to N)
                r = (i // self.grid_size) + 1
                c = (i % self.grid_size) + 1
                
                # This cell (r, c) MUST have value v
                self.cnf.append([self.vars[r][c][v]])

    def _build_cnf(self):
        """
        Calls all constraint-adding methods to build the complete CNF.
        """
        self._add_cell_constraints()
        self._add_row_constraints()
        self._add_col_constraints()
        self._add_box_constraints()
        self._add_clue_constraints()

    def _decode_model(self, model: list[int]) -> list[list[int]]:
        """
        Converts the SAT solver's model (a list of N*N*N literals)
        back into an N x N Sudoku grid of integer values.
        """
        # Create a set of positive literals for fast lookup
        positive_lits = {lit for lit in model if lit > 0}
        
        grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        for r in range(1, self.grid_size + 1):
            for c in range(1, self.grid_size + 1):
                for v in range(1, self.grid_size + 1):
                    if self.vars[r][c][v] in positive_lits:
                        grid[r - 1][c - 1] = v
                        break # Found the value for this cell
        return grid

    def solve(self) -> bool:
        """
        Solves the SAT problem.
        
        Returns:
            bool: True if a solution is found, False otherwise.
        """
        # We use Glucose3 by default, a common and fast solver
        with Solver(name=self.solver_name, bootstrap_with=self.cnf) as s:
            is_solvable = s.solve()
            if is_solvable:
                self.solution = self._decode_model(s.get_model())
            else:
                self.solution = None
            return is_solvable

    def print_solution(self):
        """
        Prints the solved Sudoku grid in a human-readable format.
        """
        if self.solution is None:
            print("No solution to print.")
            return

        print("Puzzle Solution:")
        n = self.grid_size
        box_n = self.box_size
        
        # Calculate total line width for horizontal separator
        # A line has N items, (box_n - 1) separators, and joins with " "
        # Total items in `line` list: N + (box_n - 1)
        # Total spaces from join: (N + box_n - 1) - 1
        # Total width = (items) + (spaces) = (N + box_n - 1) + (N + box_n - 2)
        separator_width = (2 * n) + (2 * box_n) - 3
        
        for r_idx, row in enumerate(self.solution):
            if r_idx % box_n == 0 and r_idx != 0:
                print("-" * separator_width) # Row separator
            
            line = []
            for c_idx, val in enumerate(row):
                if c_idx % box_n == 0 and c_idx != 0:
                    line.append("|") # Column separator
                line.append(self._val_to_char(val))
            
            print(" ".join(line))


if __name__ == "__main__":
    # 9x9 Puzzle (from original file)
    puzzle_9x9 = "57....9..........8.1.........168..4......28.9..2.9416.....2.....6.9.82.4...41.6.." # 81 chars
    print(f"Attempting to solve 9x9 puzzle:")
    try:
        solver_9x9 = SudokuSATSolver(puzzle_9x9)
        if solver_9x9.solve():
            print("\n" + "="*25 + "\n")
            solver_9x9.print_solution()
        else:
            print("This Sudoku puzzle is unsatisfiable.")
    except Exception as e:
        print(f"An error occurred: {e}")
        