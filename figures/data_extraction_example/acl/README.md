# Insert this figure into an ACL LaTeX paper

Put this entire directory under `figures/data_extraction_example/acl/` in the
same project as the main `.tex` file. The file names must be:

```
figures/data_extraction_example/acl/
  data_extraction_figure_input.tex
  paper.pdf
  design.png
  management.png
  results.png
```

`paper.pdf` is the original 14-page Jensen article. The figure reads PDF pages
1, 3, and 5 from it. The three PNGs are the screenshots in left-to-right order.

In the preamble of the main ACL file, add `tikz` and the arrow library. The ACL
template already loads `graphicx`; loading it again is harmless:

```latex
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc}
```

At the desired location in the document body, insert:

```latex
\begin{figure*}[t]
  \centering
  \resizebox{\textwidth}{!}{%
    \input{figures/data_extraction_example/acl/data_extraction_figure_input.tex}%
  }
  \caption{Evidence from different pages of a paper linked to one ground-truth record.}
  \label{fig:extraction-example}
\end{figure*}
```

Compile the main file with pdfLaTeX. Do not compile
`data_extraction_figure_input.tex` alone: it is a fragment without a document
preamble. Cite the source paper and ground-truth dataset in the figure caption
or surrounding text, as appropriate for the manuscript.
