# jev-ultrafast, vendored

`snapshot.js` is a verbatim copy of [browser-use/jev-ultrafast][upstream]'s page
reader, kept here so this repository can follow that upstream rather than
drift from it. Nothing imports it at runtime. It is the reference our own
element identity and freshness guard are written against, and the thing
`scripts/resync_vendor.py` diffs when upstream moves.

## What we took, and why

Their reader and ours are the same shape already -- one `evaluate` per page
read, returning the visible controls with their labels. The difference is what
happens to an element *between* the read and the click:

- **Identity.** Ours marked each element with a `data-oh-browser-index`
  attribute and then found it again with a selector. A re-render replaces the
  node, the attribute goes with it, and the selector waits out its 30s timeout
  before failing with a bare Playwright line. Theirs keeps a `WeakMap` from
  element to id and a `Map` back, so the read holds the node itself.
- **Freshness.** They compute a `guard` per element (role, name, value, state)
  and a page-wide `marker`, and refuse an action whose guard no longer matches
  with "Page changed since this decision. Observe again." Ours had no such
  question to ask.
- **Coverage.** Before input they resolve the element's centre and check
  `document.elementFromPoint` lands inside it, so an overlay cannot take a
  click meant for what is behind it.

Measured on Chartmetric Pilot production, 2026-09-11..18: 99 of 1,141 browser
calls failed, and 48 of those were this one class -- a stale index, 30s each,
the click never made.

We did not adopt their state shape. Ours carries a semantic outline with
`above`/`viewport`/`below` positions that the product's screenshot checks read,
and theirs does not; theirs bounds page text to the viewport, ours does not.
Nor did we adopt their agent loop or its model: that loop chooses through
`api.typesafe.ai`, a third party, and page text may not leave our own
providers.

## Following upstream

`python scripts/resync_vendor.py --check` fetches the pinned upstream file and
fails if the copy here has drifted; `--update` rewrites the copy and the pin in
`VENDOR.json`. When it reports a change, read their diff against the three
behaviours above before changing ours.

[upstream]: https://github.com/browser-use/jev-ultrafast
