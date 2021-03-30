TEX = pdflatex -shell-escape -interaction=nonstopmode -file-line-error

all : cheat_sheet.pdf

clean :
	rm -rf cheat_sheet.aux cheat_sheet.log

cleanall :
	rm -rf cheat_sheet.aux cheat_sheet.log cheat_sheet.pdf

cheat_sheet.pdf : cheat_sheet.tex
	$(TEX) cheat_sheet.tex
