# EGObox Documentation with Zola

This directory contains the Zola-based documentation site for EGObox.

## Prerequisites

### Install Zola

Zola is a fast static site generator written in Rust. Install it using one of these methods:

**Option 1: Using cargo (Rust package manager)**
```bash
cargo install zola
```

**Option 2: Download from GitHub releases**
Visit https://github.com/getzola/zola/releases and download the pre-built binary for your platform.

**Option 3: Using a package manager**
- macOS (Homebrew): `brew install zola`
- Linux (Arch): `pacman -S zola`
- Windows: Download from GitHub releases

## Project Structure

```
zola_site/
├── config.toml      # Zola configuration
├── content/         # Markdown content files
│   ├── _index.md    # Home page
│   ├── getting-started.md
│   ├── examples.md
│   ├── tutorials.md
│   └── reference.md
├── templates/       # HTML templates
│   ├── index.html   # Home page template
│   └── page.html    # Regular page template
└── static/          # Static assets (CSS, JS, images)
    ├── css/
    └── js/
```

## Development

### Serve locally

```bash
cd zola_site
zola serve
```

The site will be available at `http://127.0.0.1:1111/`

### Build for production

```bash
cd zola_site
zola build
```

The built site will be in the `public/` directory.

### Check for errors

```bash
cd zola_site
zola check
```

## Migration from MkDocs

This site was migrated from MkDocs to Zola. Key changes:

1. **Configuration**: `mkdocs.yml` → `config.toml`
2. **Content**: All markdown files migrated with Zola frontmatter (`+++`)
3. **Templates**: Custom HTML templates created (no longer using Material theme)
4. **Build command**: `mkdocs serve` → `zola serve`

## Next Steps

1. Install Zola using one of the methods above
2. Run `zola serve` to preview the site locally
3. Test all links and navigation
4. Deploy to GitHub Pages by pushing the `public/` directory

## Notes

- The code snippets that used `--8<--` (MkDocs include) now reference files directly
- Consider adding a theme like `anubis` or `terminal` from Zola's theme library for a more polished look
- The search index is enabled in the config