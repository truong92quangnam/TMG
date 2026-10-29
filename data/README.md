# DUC Dataset Setup & Data Directory Layout

This project is configured to read standard document clusters and reference summaries from NIST Document Understanding Conferences (DUC).

## Current Status in Workspace
- **DUC 2004**: Available in `DUC2004/` and symlinked to `data/duc2004/`.
  - Tasks 1 & 2: 50 document clusters with 4 human reference model summaries per cluster.
  - Can run end-to-end evaluation immediately.
- **DUC 2003, DUC 2005, DUC 2006, DUC 2007**: **Not yet provided** in the workspace.
  - Code and pipeline are fully implemented and tested.
  - To run experiments on these datasets, please place the NIST data into the directories below.

---

## Directory Layout to Provide

```
data/
├── duc2003/                           # [TODO: Place DUC 2003 data here]
│   ├── docs/
│   │   ├── cluster_001/
│   │   │   ├── doc_01.txt
│   │   │   └── ...
│   └── references/
│       ├── cluster_001/
│       │   ├── model_A.txt
│       │   └── ...
│
├── duc2004/                           # [READY: 50 clusters available]
│   ├── docs/
│   │   ├── 1001/
│   │   │   ├── D1.txt
│   │   │   └── ...
│   └── references/
│       ├── Task1001_reference1.txt
│       └── ...
│
├── duc2005/                           # [TODO: Place DUC 2005 data here]
│   ├── docs/
│   ├── topics/
│   │   ├── d301i.json (or XML topicset)
│   │   └── ...
│   └── references/
│
├── duc2006/                           # [TODO: Place DUC 2006 data here]
│   ├── docs/
│   ├── topics/
│   └── references/
│
└── duc2007/                           # [TODO: Place DUC 2007 data here]
    ├── docs/
    ├── topics/
    └── references/
```

### Topic Format for Query-Focused Tasks (DUC 2005–2007)
You can provide either the raw NIST XML topic files (e.g., `duc2005_topics.xml`) or individual JSON files in `topics/`:
```json
{
  "cluster_id": "cluster_001",
  "title": "Topic title",
  "narrative": "Detailed narrative describing user information need",
  "query": "title + narrative"
}
```
The loader automatically parses both formats.

