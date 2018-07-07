# $Id: Makefile 13573 2012-01-13 16:04:02Z uegede $
# ===============================================================================
# Purpose: simple Makefile to streamline processing latex document (just say "make all" to execute)
# Author: Tomasz Skwarnicki
# Created on: 2010-09-24
# ===============================================================================

# name of the main latex file (do not include .tex)
MAIN = main

# name of command to perform Latex (either pdflatex or latex)
LATEX = pdflatex

ifeq ($(LATEX),pdflatex)
	FIGEXT = .pdf
	MAINEXT= .pdf
	BUILDCOMMAND=rm -f $(MAIN).aux && $(LATEX) $(MAIN) && bibtex $(MAIN) && $(LATEX) $(MAIN) && $(LATEX) $(MAIN)
else
	FIGEXT = .eps
	MAINEXT= .pdf
	BUILDCOMMAND=rm -f $(MAIN).aux && $(LATEX) $(MAIN) && bibtex $(MAIN) && $(LATEX) $(MAIN) && $(LATEX) $(MAIN) && dvips -z -o $(MAIN).ps $(MAIN) && ps2pdf $(MAIN).ps && rm -f head.tmp body.tmp
endif

# list of all source files
TEXSOURCES = $(wildcard *.tex) $(wildcard *.bib)
FIGSOURCES = $(wildcard figs/*$(FIGEXT))
SOURCES    = $(TEXSOURCES) $(FIGSOURCES)

# define output (could be making .ps instead)
OUTPUT = $(MAIN)$(MAINEXT)

# prescription how to make output (your favorite commands to process latex)
# do latex twice to make sure that all cross-references are updated
$(OUTPUT): $(SOURCES) Makefile
	$(BUILDCOMMAND)

# just so we can say "make all" without knowing the output name
all: $(OUTPUT)

# remove temporary files (good idea to say "make clean" before putting things back into repository)
.PHONY : clean
clean:
	rm -f *~ *.aux *.log *.bbl *.blg *.dvi *.tmp *.out *.blg *.bbl $(MAIN)$(MAINEXT) $(MAIN).ps

# remove output file
rmout:
	rm  $(OUTPUT)
