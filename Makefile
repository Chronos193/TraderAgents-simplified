# Makefile for TraderAgents Documentation

# Variables
TEX_FILE = TraderAgents_Documentation.tex
PDF_FILE = TraderAgents_Documentation.pdf
DOC_DIR = docs

# Default target
all: pdf

# Build PDF documentation
pdf:
	@echo "Building PDF documentation..."
	cd $(DOC_DIR) && pdflatex $(TEX_FILE)
	cd $(DOC_DIR) && pdflatex $(TEX_FILE)
	@echo "PDF documentation built: $(DOC_DIR)/$(PDF_FILE)"

# Clean auxiliary files
clean:
	cd $(DOC_DIR) && rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz

# Clean all generated files including PDF
clean-all: clean
	cd $(DOC_DIR) && rm -f $(PDF_FILE)

# Quick build (single pass)
quick:
	@echo "Quick building PDF documentation..."
	cd $(DOC_DIR) && pdflatex $(TEX_FILE)
	@echo "PDF documentation built: $(DOC_DIR)/$(PDF_FILE)"

# Build with latexmk (if available)
latexmk:
	@echo "Building with latexmk..."
	cd $(DOC_DIR) && latexmk -pdf $(TEX_FILE)
	@echo "PDF documentation built: $(DOC_DIR)/$(PDF_FILE)"

# Install LaTeX packages (Ubuntu/Debian)
install-deps-ubuntu:
	sudo apt-get update
	sudo apt-get install -y texlive-full

# Install LaTeX packages (macOS with Homebrew)
install-deps-mac:
	brew install --cask mactex

# Help
help:
	@echo "Available targets:"
	@echo "  pdf          - Build PDF documentation (default)"
	@echo "  quick        - Quick build (single LaTeX pass)"
	@echo "  latexmk      - Build using latexmk"
	@echo "  clean        - Clean auxiliary files"
	@echo "  clean-all    - Clean all generated files"
	@echo "  install-deps-ubuntu - Install LaTeX on Ubuntu/Debian"
	@echo "  install-deps-mac    - Install LaTeX on macOS"
	@echo "  help         - Show this help message"

.PHONY: all pdf clean clean-all quick latexmk install-deps-ubuntu install-deps-mac help
