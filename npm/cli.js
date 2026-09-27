#!/usr/bin/env node

/**
 * Executable CLI Entry Point for npm (npx souplite)
 * Author: Muhtar Jaksilikov
 */

const { SoupLite } = require("./index");

console.log("🥣 SoupLite Node.js CLI (v0.75.0) — Author: Muhtar Jaksilikov");
console.log("Optimized for Sub-2GB RAM LLM Fine-Tuning & Diagnostics\n");

const engine = new SoupLite();
const stats = engine.getMemoryStats();

console.log("📊 System Memory Diagnostic:");
console.log(` - Total System RAM: ${stats.totalMemoryGb} GB`);
console.log(` - Available RAM:    ${stats.freeMemoryGb} GB`);
console.log(` - Sub-2GB Compliant: ${stats.withinSub2GbBudget}`);
console.log("\n🚀 For full python execution run: pip install souplite && souplite quickstart");
