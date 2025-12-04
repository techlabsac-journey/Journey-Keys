/*
TechLabs Aachen — Universal JS Checker
For Web Development Quizzes (1-5)
---------------------------------------------------------
Usage:
node wd_checker.js <quiz_id> <your_name_or_email>
Example:
node wd_checker.js 1 alice@example.com
*/

import crypto from "crypto";
import { fileURLToPath, pathToFileURL } from "url";
import path from "path";
import fs from "fs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// === QUIZ CHECKERS ===

// 1 — HTML, CSS, DOM
function check_wd_key1(name, mod) {
  let score = 0;
  const nlen = name.length;
  const htmlTrue = `<h1>Hello ${name}!</h1>`;
  const cssTrue = nlen % 2 === 0 ? "background-color: blue;" : "background-color: green;";
  const domTrue = `${name} World`;
  const combinedTrue = htmlTrue + "<style>" + cssTrue + "</style>";

  try { if (mod.createHTML(name) === htmlTrue) score++; } catch {}
  try { if (mod.createCSS(name) === cssTrue) score++; } catch {}
  try { if (mod.changeDOM(name) === domTrue) score++; } catch {}
  try { if (mod.combineHTMLCSS(name) === combinedTrue) score++; } catch {}

  return [score, 4];
}

// 2 — JS Fundamentals
function check_wd_key2(name, mod) {
  let score = 0;
  const sumASCII = [...name].reduce((a, c) => a + c.charCodeAt(0), 0);
  const reversed = [...name].reverse();
  const objTrue = { name, length: name.length, isEven: name.length % 2 === 0 };
  const descTrue = `My name has ${name.length} letters`;

  try { if (mod.sumASCII(name) === sumASCII) score++; } catch {}
  try { if (JSON.stringify(mod.reverseName(name)) === JSON.stringify(reversed)) score++; } catch {}
  try { if (JSON.stringify(mod.createUserObject(name)) === JSON.stringify(objTrue)) score++; } catch {}
  try { if (mod.describeName(name) === descTrue) score++; } catch {}

  return [score, 4];
}

// 3 — React, Vite, Router
function check_wd_key3(name, mod) {
  let score = 0;
  const compTrue = `function Welcome() { return <h1>Welcome ${name}</h1>; }`;
  const routeTrue = "<Route path='/' element={<Home />} />";
  const stateTrue = "const [count, setCount] = useState(0);";
  const buttonTrue = "<button class='btn btn-primary'>Click Me</button>";

  try { if (mod.createComponent(name) === compTrue) score++; } catch {}
  try { if (mod.createRoute() === routeTrue) score++; } catch {}
  try { if (mod.useStateSnippet() === stateTrue) score++; } catch {}
  try { if (mod.bootstrapButton() === buttonTrue) score++; } catch {}

  return [score, 4];
}

// 4 Backend — NodeJS & Express
function check_wd_key4(name, mod) {
  let score = 0;
  const routeTrue = { route: "/hello", response: `Hello ${name}` };
  const sqlCreate = "CREATE TABLE users (id INT, name TEXT);";
  const sqlInsert = `INSERT INTO users (name) VALUES ('${name}');`;
  const sqlSelect = `SELECT * FROM users WHERE name = '${name}';`;

  try { if (JSON.stringify(mod.createServer(name)) === JSON.stringify(routeTrue)) score++; } catch {}
  try { if (mod.createSQL() === sqlCreate) score++; } catch {}
  try { if (mod.insertSQL(name) === sqlInsert) score++; } catch {}
  try { if (mod.selectSQL(name) === sqlSelect) score++; } catch {}

  return [score, 4];
}

// 5 Frontend  — React Frontend Specialization
function check_wd_key5(name, mod) {
  let score = 0;
  const nlen = name.length;

  const greetingTrue = `<h2>Hello, ${name}!</h2>`;
  const hookTrue = `const [count, setCount] = useState(${nlen});`;
  const listTrue = "<ul><li>React</li><li>HTML</li><li>CSS</li></ul>";
  const buttonTrue =
    nlen % 2 === 0
      ? "<button class='btn btn-success'>Even Length</button>"
      : "<button class='btn btn-success'>Odd Length</button>";
  const fullTrue = `<div>${greetingTrue}${listTrue}${buttonTrue}</div>`;

  try { if (mod.Greeting(name) === greetingTrue) score++; } catch {}
  try { if (mod.useCountSnippet(name) === hookTrue) score++; } catch {}
  try { if (mod.renderList() === listTrue) score++; } catch {}
  try { if (mod.styledButton(name) === buttonTrue) score++; } catch {}
  try { if (mod.fullComponent(name) === fullTrue) score++; } catch {}

  return [score, 5];
}

// === QUIZ FUNCTION MAP ===
const QUIZ_FUNCTIONS = {
  1: check_wd_key1,
  2: check_wd_key2,
  3: check_wd_key3,
  4: check_wd_key4,
  5: check_wd_key5
};

// === KEY GENERATOR ===
function generateKey(name, quizId, score, total) {
  const passphrase = "techlabs2025";
  const text = `${name}:${quizId}:${score}/${total}:${passphrase}`;
  const hash = crypto.createHash("sha256").update(text).digest("hex");
  return hash.slice(0, 16); // 16-char unique key
}

// === MAIN EXECUTION ===
async function main() {
  const args = process.argv.slice(2);
  if (args.length < 2) {
    console.log("Usage: node wd_checker.js <quiz_id> <name_or_email>");
    console.log("Example: node wd_checker.js 1 alice@example.com");
    process.exit(1);
  }

  const quizId = parseInt(args[0]);
  const name = args[1];

  // FIXED: Correct file path construction with backward compatibility
  // Support both new naming (wd_key4.js, wd_key5.js) and legacy naming (wd_back_4.js, wd_front_4.js)
  let quizFile = path.join(__dirname, `wd_key${quizId}.js`);
  
  // Check if file exists, if not try legacy naming for quiz 4 and 5
  if (!fs.existsSync(quizFile)) {
    if (quizId === 4) {
      const legacyFile = path.join(__dirname, "wd_back_4.js");
      if (fs.existsSync(legacyFile)) {
        quizFile = legacyFile;
      }
    } else if (quizId === 5) {
      const legacyFile = path.join(__dirname, "wd_front_4.js");
      if (fs.existsSync(legacyFile)) {
        quizFile = legacyFile;
      }
    }
  }
  
  if (!fs.existsSync(quizFile)) {
    console.log(`❌ Quiz file not found: ${quizFile}`);
    if (quizId === 4) {
      console.log(`Make sure either wd_key4.js or wd_back_4.js exists in the same directory.`);
    } else if (quizId === 5) {
      console.log(`Make sure either wd_key5.js or wd_front_4.js exists in the same directory.`);
    } else {
      console.log(`Make sure the file wd_key${quizId}.js exists in the same directory.`);
    }
    process.exit(1);
  }

  const mod = await import(pathToFileURL(path.resolve(quizFile)).href);
  const checker = QUIZ_FUNCTIONS[quizId];
  
  if (!checker) {
    console.log(`❌ Unknown quiz ID: ${quizId}`);
    console.log("Valid quiz IDs are: 1, 2, 3, 4, 5");
    process.exit(1);
  }

  const [score, total] = checker(name, mod);
  const key = generateKey(name, quizId, score, total);

  console.log(`✅ You scored ${score}/${total}`);
  console.log(`🔑 Your unique key: ${key}`);
}

main();