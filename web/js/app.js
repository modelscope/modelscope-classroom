/* ===== State ===== */
let contentIndex = null;
let allFiles = [];     // flat list of {file, title, chapterTitle} for search & nav

// Base path for fetching lecture content files.
// When served from web/ subdir locally, need '../' to reach repo root.
// When deployed to GitHub Pages (web/ is root with content copied in), use ''.
const CONTENT_BASE = (location.hostname === 'localhost' || location.hostname === '127.0.0.1') ? '../' : '';

/* ===== Init ===== */
document.addEventListener('DOMContentLoaded', async () => {
    mermaid.initialize({ startOnLoad: false, theme: 'default', securityLevel: 'loose' });

    try {
        const resp = await fetch('content-index.json');
        contentIndex = await resp.json();
        buildFileList(contentIndex.chapters);
        renderSidebar(contentIndex.chapters);
        handleRoute();
    } catch (e) {
        document.getElementById('content').innerHTML =
            '<div class="error-msg">Failed to load content index. Run <code>python build_index.py</code> first.</div>';
    }

    window.addEventListener('hashchange', handleRoute);
    setupUI();
});

/* ===== Build flat file list for prev/next navigation ===== */
function buildFileList(chapters) {
    allFiles = [];
    for (const ch of chapters) {
        collectFiles(ch.children, ch.title);
    }
}
function collectFiles(children, chapterTitle) {
    for (const c of children) {
        if (c.type === 'file') {
            allFiles.push({ file: c.file, title: c.title, chapterTitle });
        } else if (c.children) {
            collectFiles(c.children, chapterTitle);
        }
    }
}

/* ===== Routing ===== */
function handleRoute() {
    const hash = decodeURIComponent(window.location.hash.slice(1));
    if (hash && hash.endsWith('.md')) {
        loadContent(hash);
    } else {
        renderWelcome();
    }
}

/* ===== Sidebar Rendering ===== */
function renderSidebar(chapters) {
    const nav = document.getElementById('sidebarNav');
    nav.innerHTML = chapters.map(ch => {
        const childrenHtml = renderNavChildren(ch.children);
        return `
            <div class="nav-chapter" data-id="${ch.id}">
                <div class="nav-chapter-header" onclick="toggleChapter(this)">
                    <span class="nav-chapter-num">${ch.id}</span>
                    <span class="nav-chapter-label">${ch.title}</span>
                    <svg class="nav-arrow" width="14" height="14" viewBox="0 0 24 24" fill="none"
                         stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"/></svg>
                </div>
                <div class="nav-children">${childrenHtml}</div>
            </div>`;
    }).join('');
}

function renderNavChildren(children) {
    return children.map(c => {
        if (c.type === 'file') {
            return `<a class="nav-item" href="#${encodeURIComponent(c.file)}"
                       data-file="${c.file}" title="${c.title}">${c.title}</a>`;
        }
        if (c.type === 'group') {
            return `
                <div class="nav-group open">
                    <div class="nav-group-title" onclick="toggleGroup(this)">${c.title}</div>
                    <div class="nav-group-children">${renderNavChildren(c.children)}</div>
                </div>`;
        }
        return '';
    }).join('');
}

/* ===== Sidebar Toggle ===== */
function toggleChapter(el) {
    const chapter = el.closest('.nav-chapter');
    chapter.classList.toggle('open');
}
function toggleGroup(el) {
    el.closest('.nav-group').classList.toggle('open');
}

/* ===== Welcome Page ===== */
function renderWelcome() {
    if (!contentIndex) return;
    const chs = contentIndex.chapters;
    const totalFiles = allFiles.length;

    document.getElementById('content').innerHTML = `
        <div class="welcome-hero">
            <img class="welcome-banner" src="${CONTENT_BASE}大模型讲义/title_image.png" alt="大模型技术基础：数学、训练与智能体">
            <h1>大模型讲义</h1>
            <p>涵盖完整的大模型理论与实践课程，从深度学习基础到智能体前沿技术</p>
            <div class="welcome-stats">
                <div><span>${chs.length}</span> 章节</div>
                <div><span>${totalFiles}</span> 篇文章</div>
            </div>
        </div>
        <div class="chapter-grid">
            ${chs.map(ch => `
                <div class="chapter-card" onclick="openChapterFirst('${ch.id}')">
                    <div class="chapter-card-num">${ch.id}</div>
                    <div class="chapter-card-title">${ch.title}</div>
                    <div class="chapter-card-count">${ch.count} 节</div>
                </div>`).join('')}
        </div>`;

    updateActiveNav(null);
    document.title = '大模型讲义 — ModelScope Classroom';
}

function openChapterFirst(chapterId) {
    const ch = contentIndex.chapters.find(c => c.id === chapterId);
    if (!ch) return;
    const first = findFirstFile(ch.children);
    if (first) {
        window.location.hash = encodeURIComponent(first.file);
    }
}
function findFirstFile(children) {
    for (const c of children) {
        if (c.type === 'file') return c;
        if (c.children) {
            const f = findFirstFile(c.children);
            if (f) return f;
        }
    }
    return null;
}

/* ===== Load & Render Content ===== */
async function loadContent(filePath) {
    const contentEl = document.getElementById('content');
    contentEl.innerHTML = '<div class="loading">加载中...</div>';

    try {
        const resp = await fetch(CONTENT_BASE + filePath);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        let md = await resp.text();
        const basePath = filePath.substring(0, filePath.lastIndexOf('/') + 1);

        // Build breadcrumb
        const crumbs = buildBreadcrumb(filePath);

        // Protect math from markdown parser
        const { text: safeText, blocks: mathBlocks } = protectMath(md);

        // Parse markdown
        let html = marked.parse(safeText, { gfm: true, breaks: false });

        // Restore math expressions
        html = restoreMath(html, mathBlocks);

        // Prev/Next nav
        const navHtml = buildArticleNav(filePath);

        contentEl.innerHTML = `${crumbs}<article class="markdown-body">${html}</article>${navHtml}`;

        const article = contentEl.querySelector('.markdown-body');

        // Fix relative image paths
        article.querySelectorAll('img').forEach(img => {
            const src = img.getAttribute('src');
            if (src && !src.startsWith('http') && !src.startsWith('data:')) {
                img.src = CONTENT_BASE + basePath + src;
            }
        });

        // Fix relative links (to other .md files)
        article.querySelectorAll('a[href]').forEach(a => {
            const href = a.getAttribute('href');
            if (href && href.endsWith('.md') && !href.startsWith('http')) {
                // Resolve relative path
                const resolved = resolveRelPath(basePath, href);
                a.href = '#' + encodeURIComponent(resolved);
            }
        });

        // Syntax highlight
        article.querySelectorAll('pre code').forEach(block => {
            if (!block.classList.contains('language-mermaid')) {
                hljs.highlightElement(block);
            }
        });

        // Mermaid
        const mermaidCodes = article.querySelectorAll('code.language-mermaid');
        let mermaidId = 0;
        mermaidCodes.forEach(code => {
            const pre = code.parentElement;
            const div = document.createElement('div');
            div.className = 'mermaid';
            div.id = 'mermaid-' + (mermaidId++);
            div.textContent = code.textContent;
            pre.replaceWith(div);
        });
        if (mermaidCodes.length > 0) {
            try { await mermaid.run({ nodes: article.querySelectorAll('.mermaid') }); } catch (_) {}
        }

        // KaTeX math rendering
        if (typeof renderMathInElement === 'function') {
            renderMathInElement(article, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\[', right: '\\]', display: true },
                    { left: '\\(', right: '\\)', display: false },
                ],
                throwOnError: false,
            });
        }

        // Update UI
        updateActiveNav(filePath);
        expandToFile(filePath);
        contentEl.scrollTo(0, 0);

        // Update title
        const h1 = article.querySelector('h1');
        document.title = (h1 ? h1.textContent + ' — ' : '') + '大模型讲义';

    } catch (err) {
        contentEl.innerHTML = `<div class="error-msg">加载失败：${err.message}<br><br>
            <a href="#" onclick="window.location.hash='';renderWelcome();return false;">返回首页</a></div>`;
    }

    // Close mobile sidebar
    document.getElementById('sidebar').classList.remove('open');
    document.getElementById('sidebarOverlay').classList.remove('open');
}

/* ===== Resolve relative path ===== */
function resolveRelPath(base, rel) {
    const parts = base.split('/').filter(Boolean);
    const relParts = rel.split('/');
    for (const p of relParts) {
        if (p === '..') parts.pop();
        else if (p !== '.') parts.push(p);
    }
    return parts.join('/');
}

/* ===== Breadcrumb ===== */
function buildBreadcrumb(filePath) {
    const info = findFileInfo(filePath);
    if (!info) return '';
    return `<div class="breadcrumb">
        <a href="#" onclick="window.location.hash='';renderWelcome();return false;">首页</a>
        <span class="breadcrumb-sep">›</span>
        <span>${info.chapterTitle}</span>
        <span class="breadcrumb-sep">›</span>
        <span>${info.title}</span>
    </div>`;
}

function findFileInfo(filePath) {
    return allFiles.find(f => f.file === filePath);
}

/* ===== Prev/Next Navigation ===== */
function buildArticleNav(filePath) {
    const idx = allFiles.findIndex(f => f.file === filePath);
    if (idx < 0) return '';
    const prev = idx > 0 ? allFiles[idx - 1] : null;
    const next = idx < allFiles.length - 1 ? allFiles[idx + 1] : null;

    let html = '<div class="article-nav">';
    if (prev) {
        html += `<a class="prev" href="#${encodeURIComponent(prev.file)}">
            <div class="article-nav-label">← 上一篇</div>
            <div class="article-nav-title">${prev.title}</div></a>`;
    } else {
        html += '<div></div>';
    }
    if (next) {
        html += `<a class="next" href="#${encodeURIComponent(next.file)}">
            <div class="article-nav-label">下一篇 →</div>
            <div class="article-nav-title">${next.title}</div></a>`;
    }
    html += '</div>';
    return html;
}

/* ===== Active Nav Highlight ===== */
function updateActiveNav(filePath) {
    document.querySelectorAll('.nav-item').forEach(el => {
        el.classList.toggle('active', el.dataset.file === filePath);
    });
    // Highlight parent chapter header
    document.querySelectorAll('.nav-chapter-header').forEach(el => el.classList.remove('active'));
    if (filePath) {
        const activeItem = document.querySelector(`.nav-item[data-file="${filePath}"]`);
        if (activeItem) {
            const chapter = activeItem.closest('.nav-chapter');
            if (chapter) chapter.querySelector('.nav-chapter-header')?.classList.add('active');
        }
    }
}

function expandToFile(filePath) {
    const item = document.querySelector(`.nav-item[data-file="${filePath}"]`);
    if (!item) return;
    // Open parent chapter
    const chapter = item.closest('.nav-chapter');
    if (chapter) chapter.classList.add('open');
    // Open parent group
    const group = item.closest('.nav-group');
    if (group) group.classList.add('open');
    // Scroll into view
    item.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

/* ===== UI Setup ===== */
function setupUI() {
    // Mobile sidebar toggle
    document.getElementById('menuToggle').addEventListener('click', () => {
        document.getElementById('sidebar').classList.toggle('open');
        document.getElementById('sidebarOverlay').classList.toggle('open');
    });
    document.getElementById('sidebarOverlay').addEventListener('click', () => {
        document.getElementById('sidebar').classList.remove('open');
        document.getElementById('sidebarOverlay').classList.remove('open');
    });

    // Back to top
    const btn = document.getElementById('backToTop');
    const contentEl = document.getElementById('content');
    window.addEventListener('scroll', () => {
        btn.classList.toggle('visible', window.scrollY > 400);
    }, { passive: true });
    btn.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // Search
    document.getElementById('searchInput').addEventListener('input', (e) => {
        const q = e.target.value.trim().toLowerCase();
        filterSidebar(q);
    });
}

/* ===== Search Filter ===== */
/* ===== Math Protection (prevent marked from mangling LaTeX) ===== */
function protectMath(text) {
    const blocks = [];
    // Protect fenced math in code blocks — skip those
    // Protect display math $$...$$
    text = text.replace(/\$\$([\s\S]+?)\$\$/g, (match) => {
        const id = blocks.length;
        blocks.push(match);
        return `\n\nMATH_PLACEHOLDER_${id}\n\n`;
    });
    // Protect inline math $...$  (not inside code spans)
    text = text.replace(/(?<!\$|`)\$(?!\$)([^\$\n]+?)\$(?!\$|`)/g, (match) => {
        const id = blocks.length;
        blocks.push(match);
        return `MATH_PLACEHOLDER_${id}`;
    });
    return { text, blocks };
}

function restoreMath(html, blocks) {
    for (let i = 0; i < blocks.length; i++) {
        const placeholder = `MATH_PLACEHOLDER_${i}`;
        // marked may wrap block placeholders in <p> tags
        html = html.replace(`<p>${placeholder}</p>`, blocks[i]);
        html = html.replace(placeholder, blocks[i]);
    }
    return html;
}

function filterSidebar(query) {
    const chapters = document.querySelectorAll('.nav-chapter');
    chapters.forEach(ch => {
        const items = ch.querySelectorAll('.nav-item');
        let hasMatch = false;
        items.forEach(item => {
            const title = (item.textContent || '').toLowerCase();
            const match = !query || title.includes(query);
            item.style.display = match ? '' : 'none';
            if (match) hasMatch = true;
        });
        // Also check chapter title
        const label = ch.querySelector('.nav-chapter-label');
        if (label && label.textContent.toLowerCase().includes(query)) hasMatch = true;

        ch.style.display = hasMatch ? '' : 'none';
        if (query && hasMatch) ch.classList.add('open');
    });
}
