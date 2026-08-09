# Juice Shop Loop 12 catalog v2 rejection

Catalog v2 correctly inspected evaluator descriptions for external dependencies, but its short-abbreviation substring filter treated `rce` inside the ordinary word `enforce` as remote-code execution. This incorrectly removed a local redirect task whose HTTP response can be evaluated without following its external `Location`.

No model or policy result was run on catalog v2. Catalog v3 replaces short substring matching with explicit dangerous challenge-key prefixes and full dangerous phrases. The target browser and HTTP client remain configured not to follow redirects.
