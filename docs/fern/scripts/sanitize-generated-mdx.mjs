// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { readdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

const root =
  process.argv[2] ?? "product-docs/molt/Full-Library-Reference";

const escapeTemplate = (value) =>
  value
    .replaceAll("\\", "\\\\")
    .replaceAll("`", "\\`")
    .replaceAll("${", "\\${");

async function* mdxFiles(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      yield* mdxFiles(path);
    } else if (entry.isFile() && path.endsWith(".mdx")) {
      yield path;
    }
  }
}

let changed = 0;

for await (const file of mdxFiles(root)) {
  const before = await readFile(file, "utf8");
  const after = before
    .split("\n")
    .map((line) => {
      if (!line.includes("<ParamField") || !line.includes(' type="')) {
        return line;
      }

      const match = line.match(/^(.*\stype=")(.*)(">\s*)$/);
      if (match == null) {
        return line;
      }

      const left = match[1].slice(0, -1);
      const value = escapeTemplate(match[2]);
      return `${left}{\`${value}\`}>`;
    })
    .join("\n");

  if (after !== before) {
    await writeFile(file, after);
    changed += 1;
  }
}

console.log(`Sanitized generated MDX in ${changed} file(s).`);
