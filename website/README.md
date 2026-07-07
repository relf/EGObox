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
website/
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
cd website
zola serve
```

The site will be available at `http://127.0.0.1:1111/`

### Build for production

```bash
cd website
zola build
```

The built site will be in the `public/` directory.

### Check for errors

```bash
cd website
zola check
```
