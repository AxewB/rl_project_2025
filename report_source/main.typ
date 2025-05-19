
#import "settings/settings.typ": settings
#import "settings/settings.typ": img
#import "settings/appendix.typ": appendix
#set par(leading: 1.5em)
#show: settings.with()

#include "title.typ"
#include "problem.typ"
// #include "summary.typ"

#set page(numbering: "1")

#outline(title: "Содержание")

#pagebreak()

#include "content.typ"

#include "appendix.typ"
