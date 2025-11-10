// To run the solver, use npm install then npm start.
import fs from "fs";
import csv from "csv-parser";

import SudokuGenerator from './SudokuGenerator.js';
import INITIAL_STATE from './INITIAL_STATE.js';
import SudokuRunner from "./SudokuRunner.js";

/**This block works with the original implementation
// const generator = new SudokuGenerator();
// const easyPuzzle = generator.generateEasyPuzzle();

// await generator.fetchAndSolveEachPuzzle("test_sudoku.csv");

// console.log('Generated Easy Puzzle:');
// console.log(easyPuzzle);

/** This function works with the 
// async function main() {
//     const generator = new SudokuGenerator();
//     await generator.fetchAndSolveEachPuzzle("test-sudoku.csv");
// }
*/

async function main() {
  const runner = new SudokuRunner();
  // Solve sudokus and save the profile results in csv
  await runner.processCSV("sudoku_test_set_random_10k", "sudoku_results.csv");
}

main().catch(console.error);