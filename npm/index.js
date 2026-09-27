/**
 * SoupLite npm Node.js Binding
 * Author: Muhtar Jaksilikov
 * Ultra-lightweight LLM Fine-Tuning & Inference engine optimized for sub-2GB RAM.
 */

const { execSync } = require("child_process");
const os = require("os");

class SoupLite {
  constructor(options = {}) {
    this.maxMemoryMb = options.maxMemoryMb || 2048;
  }

  /**
   * Run RAM Memory Diagnostic Benchmark
   */
  getMemoryStats() {
    const totalMemGb = (os.totalmem() / (1024 * 1024 * 1024)).toFixed(2);
    const freeMemGb = (os.freemem() / (1024 * 1024 * 1024)).toFixed(2);
    return {
      totalMemoryGb: parseFloat(totalMemGb),
      freeMemoryGb: parseFloat(freeMemGb),
      withinSub2GbBudget: parseFloat(totalMemGb) <= 2.0,
      status: "ONLINE",
    };
  }

  /**
   * Execute CLI command via python souplite
   */
  runCLI(args = []) {
    try {
      const output = execSync(`souplite ${args.join(" ")}`, { encoding: "utf-8" });
      return output;
    } catch (err) {
      return `SoupLite CLI error: ${err.message}`;
    }
  }
}

module.exports = {
  SoupLite,
  version: "0.75.0",
  author: "Muhtar Jaksilikov",
};

if (process.argv.includes("--test")) {
  const engine = new SoupLite();
  console.log("SoupLite npm Engine initialized successfully!");
  console.log("Stats:", engine.getMemoryStats());
}
