# Juice Shop Loop 12 catalog v1 rejection

The first mechanically generated catalog contained 25 tasks, but its safety filter inspected only the challenge key and name. Evaluator-only descriptions revealed tasks that require Internet-derived data, Google/OAuth interaction, Pastebin-derived information, or an external SoundCloud payload.

This violates the preregistered exclusion of real external dependencies. Catalog v1 is retained as rejected evidence and must not be used for baseline or training. Catalog v2 applies the same model-outcome-independent ordering after extending the safety filter to evaluator descriptions. No model result was observed before this correction.
