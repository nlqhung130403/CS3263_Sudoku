// SudokuRunner.ts
import fs from "fs";
import csv from "csv-parser";
import { performance } from "perf_hooks";
import SudokuGenerator from "./SudokuGenerator.js";

interface PuzzleRow {
  id: string;
  puzzle: string;
  solution: string;
  clues: string;
  difficulty: string;
}

interface ResultRow {
  id: string;
  puzzle: string;
  solution: string;
  time_sec: string;
  difficulty: string;
}

class SudokuRunner {
  private generator: SudokuGenerator;

  constructor() {
    this.generator = new SudokuGenerator();
  }

  /** Solve a single puzzle and return results */
  public async fetchSolvePuzzle(row: PuzzleRow): Promise<ResultRow> {
    const puzzleStr = row.puzzle;

    // Convert puzzle string to 9x9 grid
    const grid: number[][] = [];
    for (let i = 0; i < 9; i++) {
      const rowArr: number[] = [];
      for (let j = 0; j < 9; j++) {
        const char = puzzleStr[i * 9 + j];
        rowArr.push(char === "." ? 0 : parseInt(char, 10));
      }
      grid.push(rowArr);
    }

    // Set the generator grid
    this.generator.setGrid(grid);

    // Measure time for solving
    const startTime = performance.now();
    await this.generator.solveCurrentPuzzle();
    const endTime = performance.now();
    const elapsed = ((endTime - startTime) / 1000).toFixed(3);

    // Flatten solved grid to string
    const solvedGridStr = this.generator.getGrid().flat().join("");

    return {
      id: row.id,
      puzzle: puzzleStr,
      solution: solvedGridStr,
      time_sec: elapsed,
      difficulty: row.difficulty,
    };
  }

  /** Process all puzzles from input CSV and save results */
  public async processCSV(inputCsv: string, outputCsv: string) {
    // Write CSV header
    const header = "id,puzzle,solution,time_sec,difficulty\n";
    fs.writeFileSync(outputCsv, header);

    return new Promise<void>((resolve, reject) => {
      const stream = fs.createReadStream(inputCsv).pipe(csv());

      stream.on("data", async (row: PuzzleRow) => {
        // Pause the stream while processing async
        stream.pause();

        try {
          const result = await this.fetchSolvePuzzle(row);
          const line = `${result.id},${result.puzzle},${result.solution},${result.time_sec},${result.difficulty}\n`;
          fs.appendFileSync(outputCsv, line);
        } catch (err) {
          console.error("Error solving puzzle ID:", row.id, err);
        }

        // Resume stream after processing
        stream.resume();
      });

      stream.on("end", () => {
        console.log(`✅ Finished processing. Results saved to ${outputCsv}`);
        resolve();
      });

      stream.on("error", (err) => {
        reject(err);
      });
    });
  }
}

export default SudokuRunner;