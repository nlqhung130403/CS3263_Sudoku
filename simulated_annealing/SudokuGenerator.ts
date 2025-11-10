/**
 * @file SudokuGenerator.ts
 * @original-author dsbalico — https://github.com/dsbalico/sudoku-solver-using-simulated-annealing/blob/main/src/algorithm/SudokuGenerator.ts
 * @description Generates Sudoku puzzles and solves them using SudokuSolver.
 */

import { performance } from "perf_hooks";
import INITIAL_STATE from "./INITIAL_STATE";
import SudokuSolver from "./SudokuSolver";
import fs from "fs";
import csv from "csv-parser";


class SudokuGenerator {
    private grid: number[][];

    constructor() {
        this.grid = [];
        for (let i = 0; i < 9; i++) {
            this.grid[i] = [];
            for (let j = 0; j < 9; j++) {
                this.grid[i][j] = 0;
            }
        }
    }

    private clearBoard(): void {
        for(let y = 0; y < 9; y++) {
            for(let x = 0; x < 9; x++) {
                this.grid[y][x] = 0;
            }
        }
    }

    private solveSudoku(): void {
        console.log("Solving sudoku from generator.");
        const solver = new SudokuSolver(this.grid, INITIAL_STATE.maxIter, 
                                        INITIAL_STATE.initTemp, 
                                        INITIAL_STATE.coolingRate, 
                                        INITIAL_STATE.reheatTo, 
                                        INITIAL_STATE.reheatForX);
        let isSolved = false;
        
        // Run the solver until the sudoku is solved.
        while (!isSolved) {
            const result = solver.solveSudoku();
    
            if (result[1] === true) {
                this.grid = result[0];
                isSolved = true; // Set the flag to exit the loop
            }
        }
    }
    

    private removeCells(cellsToRemove: number): void {
        let count = 0;

        while (count < cellsToRemove) {
            const row = Math.floor(Math.random() * 9);
            const col = Math.floor(Math.random() * 9);
            if (this.grid[row][col] !== 0) {
                this.grid[row][col] = 0;
                count++;
            }
        }
    }

    public generateEasyPuzzle(): number[][] {
        this.clearBoard();
        this.solveSudoku();
        this.removeCells(20);
        console.log("[Easy] 20 numbers removed.");
        return this.grid;
    }

    public generateMediumPuzzle(): number[][] {
        this.clearBoard();
        this.solveSudoku();
        this.removeCells(45);
        console.log("[Medium] 45 numbers removed.");
        return this.grid;
    }

    public generateHardPuzzle(): number[][] {
        this.clearBoard();
        this.solveSudoku();
        this.removeCells(61);
        console.log("[Hard] 61 numbers removed.");
        return this.grid;
    }

    /** Process puzzles row by row as they are read */
    public fetchAndSolveEachPuzzle(csvPath: string = "sudoku-3m.csv", maxRows: number = 10): void {
        let rowCount = 0;

        fs.createReadStream(csvPath)
            .pipe(csv())
            .on("data", (data) => {
                if (rowCount >= maxRows) return;

                const puzzleStr = data.puzzle;
                const grid: number[][] = [];

                for (let i = 0; i < 9; i++) {
                    const row: number[] = [];
                    for (let j = 0; j < 9; j++) {
                        const char = puzzleStr[i * 9 + j];
                        row.push(char === "." ? 0 : parseInt(char, 10));
                    }
                    grid.push(row);
                }

                // Set the current grid
                this.grid = grid;

                // Time the private solveSudoku
                // const startTime = performance.now();
                // Solve immediately inside on("data")
                this.solveSudoku();
                // const endTime = performance.now();
                // const elapsed = ((endTime - startTime) / 1000).toFixed(3);

                // const solvedGridStr = this.grid.flat().join("");
                console.log(`Solved puzzle ID ${data.id}:`);
                console.table(this.grid);

                rowCount++;
            })
            .on("end", () => {
                console.log(`Finished processing ${rowCount} puzzles.`);
            })
            .on("error", (err) => {
                console.error("Error reading CSV:", err);
            });
    }

    // setter for an external grid
    public setGrid(grid: number[][]) {
        this.grid = grid;
    }

    // getter for an external grid
    public getGrid() {
        return this.grid;
    }

    // public async wrapper around the private solve
    public async solveCurrentPuzzle(): Promise<void> {
        // wrap the private solveSudoku to make it awaitable
        return new Promise((resolve) => {
            this.solveSudoku(); // private function inside the class
            resolve();
        });
    }
}

export default SudokuGenerator