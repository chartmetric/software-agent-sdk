SEMANTIC_OUTLINE_SCRIPT = r"""
() => {
  const LIMIT = 80;
  const selector = [
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', '[role="heading"]',
    'main', 'nav', 'aside', 'section', 'article', 'form',
    '[role="main"]', '[role="navigation"]', '[role="complementary"]',
    '[role="region"]', '[role="form"]'
  ].join(',');
  const rendered = (element) => {
    if (element.getClientRects().length === 0) return false;
    for (let node = element; node; node = node.parentElement) {
      const style = window.getComputedStyle(node);
      if (style.visibility === 'hidden' || style.display === 'none') return false;
      if (Number.parseFloat(style.opacity || '1') <= 0.05) return false;
    }
    return true;
  };
  const text = (element) => {
    if (!element) return '';
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
    const parts = [];
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      if (node.parentElement && rendered(node.parentElement)) {
        const value = (node.nodeValue || '').trim();
        if (value) parts.push(value);
      }
    }
    return parts.join(' ').replace(/\s+/g, ' ').slice(0, 160);
  };
  const location = (rect) => {
    if (rect.bottom < 0) return 'above';
    if (rect.top > window.innerHeight) return 'below';
    return 'viewport';
  };
  const labelledName = (element) => {
    const ariaLabel = (element.getAttribute('aria-label') || '').trim();
    if (ariaLabel) return ariaLabel;
    const labelledBy = element.getAttribute('aria-labelledby');
    if (labelledBy) {
      const label = document.getElementById(labelledBy);
      const labelText = text(label);
      if (labelText) return labelText;
    }
    const heading = element.querySelector(
      ':scope > h1, :scope > h2, :scope > h3, :scope > h4, :scope > h5, :scope > h6'
    ) || element.querySelector('h1, h2, h3, h4, h5, h6');
    return text(heading);
  };

  const all = [];
  for (const element of document.querySelectorAll(selector)) {
    if (!rendered(element)) continue;
    const rect = element.getBoundingClientRect();
    const tag = element.tagName.toLowerCase();
    const explicitRole = (element.getAttribute('role') || '').toLowerCase();
    const isHeading = /^h[1-6]$/.test(tag) || explicitRole === 'heading';
    const name = isHeading ? text(element) : labelledName(element);
    // A landmark with neither a name nor an id says nothing worth a row. One
    // with an id says plenty even when it has no name yet -- that is exactly
    // the deferred section: its heading lives inside the part that has not
    // rendered, so it is nameless *because* it has not mounted, and dropping it
    // here hides the one handle that would reach it. `browser_scroll to_id`
    // can go straight to it; nothing can go to a row that was never listed.
    if (!name && !element.id && !['main', 'nav', 'aside'].includes(tag)) continue;
    const role = explicitRole || (
      tag === 'nav' ? 'navigation'
      : tag === 'aside' ? 'complementary'
      : tag === 'main' ? 'main'
      : tag
    );
    // For a nameless-but-identified container, the role alone ("section") tells
    // a reader nothing and reads like a row not worth acting on. Naming it by
    // its id, and saying it is empty, is what makes it actionable.
    const inferredName = name || (element.id ? '#' + element.id + ' (empty)' : role);
    all.push({
      kind: isHeading ? 'heading' : 'landmark',
      tag,
      role,
      level: isHeading
        ? Number.parseInt(
          element.getAttribute('aria-level') || tag.slice(1), 10
        ) || null
        : null,
      name: inferredName.slice(0, 160),
      id: (element.id || '').slice(0, 120),
      y: Math.round(rect.top + window.scrollY),
      location: location(rect),
    });
  }
  return {items: all.slice(0, LIMIT), total: all.length, truncated: all.length > LIMIT};
}
"""


FIND_VISIBLE_TEXT_SCRIPT = r"""
({needle, limit}) => {
  const wanted = needle.trim().toLowerCase();
  if (!wanted) return {query: needle, matches: [], truncated: false};
  const rendered = (element) => {
    if (element.getClientRects().length === 0) return false;
    for (let node = element; node; node = node.parentElement) {
      const style = window.getComputedStyle(node);
      if (style.visibility === 'hidden' || style.display === 'none') return false;
      if (Number.parseFloat(style.opacity || '1') <= 0.05) return false;
    }
    return true;
  };
  const normalizedText = (element) => {
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
    const parts = [];
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      if (node.parentElement && rendered(node.parentElement)) {
        const value = (node.nodeValue || '').trim();
        if (value) parts.push(value);
      }
    }
    return parts.join(' ').replace(/\s+/g, ' ');
  };
  const elements = Array.from(document.body.querySelectorAll('*'));
  const matchedAncestors = new WeakSet();
  const deepest = [];
  let truncated = false;
  for (let index = elements.length - 1; index >= 0; index -= 1) {
    const element = elements[index];
    if (matchedAncestors.has(element)) continue;
    if (['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(element.tagName)) continue;
    if (!(element.textContent || '').toLowerCase().includes(wanted)) continue;
    if (!rendered(element)) continue;
    if (!normalizedText(element).toLowerCase().includes(wanted)) continue;
    if (deepest.length === limit) {
      truncated = true;
      break;
    }
    deepest.push(element);
    for (let parent = element.parentElement; parent; parent = parent.parentElement) {
      matchedAncestors.add(parent);
    }
  }
  deepest.reverse();
  const headings = Array.from(
    document.querySelectorAll('h1,h2,h3,h4,h5,h6,[role="heading"]')
  ).filter(rendered);
  const location = (rect) => {
    if (rect.bottom < 0) return 'above';
    if (rect.top > window.innerHeight) return 'below';
    return 'viewport';
  };
  const headingFor = (element) => {
    let closest = '';
    for (const heading of headings) {
      if (heading === element || heading.contains(element)) {
        closest = normalizedText(heading);
        break;
      }
      if (heading.compareDocumentPosition(element) & Node.DOCUMENT_POSITION_FOLLOWING) {
        closest = normalizedText(heading);
      }
    }
    return closest.slice(0, 160);
  };
  const matches = deepest.map((element) => {
    const rect = element.getBoundingClientRect();
    return {
      tag: element.tagName.toLowerCase(),
      role: (element.getAttribute('role') || '').slice(0, 80),
      text: normalizedText(element).slice(0, 240),
      id: (element.id || '').slice(0, 120),
      y: Math.round(rect.top + window.scrollY),
      location: location(rect),
      heading: headingFor(element),
    };
  });
  return {
    query: needle,
    matches,
    truncated,
  };
}
"""
