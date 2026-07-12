# `simple_brats` interactive explainer

Open [`index.html`](./index.html) directly in a browser. It has no network or build dependencies;
all styles, interaction logic, and diagrams are embedded, and the small MRI derivatives live in
`assets/`.

For the most reliable local behavior, serve the repository root and open the page over HTTP:

```bash
python3 -m http.server 8765
```

Then visit `http://127.0.0.1:8765/docs/repo-explainer/`.

The explainer is an audit-oriented map of the current implementation. It deliberately distinguishes
implemented behavior, registered-but-unimplemented experiments, and documentation/code conflicts.
Code links use a pinned GitHub snapshot SHA so that line references remain stable.
