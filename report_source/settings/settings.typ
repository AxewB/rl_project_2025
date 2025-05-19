#import "figures.typ": figures
#import "lists.typ": lists
#import "headings.typ": headings

#let settings(
  doc,
) = {
  set page(
    paper: "a4",
    margin: (top:2cm, bottom:2cm, left:3cm, right:1cm),
  )



  set align(center)
  set text(font: "Times New Roman", 14pt)

  show: figures.with()
  show: lists.with()
  show: headings.with()

  // Стилизация листингов
  // show raw: content => {
  //   set text(size: 10pt, font: "Cousine")
  //   set par(leading: 0.5em)
  //   content
  // }


  set ref(supplement: "")

  show figure.where(
    kind: table
  ): tb => {
    set figure.caption(position: top);
    show figure.caption: set align(left)

    tb
  }
  show figure.where(
    kind: table
  ): set figure(supplement: "Таблица")

  show figure.where(
    kind: image
  ): set figure(supplement: "Рисунок")


  show outline.entry: entry => {
    let body = lower(entry.body())
    let isAppendix = lower(repr(repr(body.child))).contains("приложение")


    if isAppendix {
      set par(first-line-indent: 0pt)
      let prefix = entry.prefix()
      let page = entry.page()

      pad(left: 0pt,
        [
          #"Приложение"
          #prefix
          #box(width: 1fr, repeat([.], gap: 0.15em))
          #page
        ]
      )
    } else {
      entry
    }
  }

  set par(justify: true, leading: 1em, first-line-indent: (amount: 1.25cm, all: true))
  set align(left)

  doc
}

#let img(source, caption, width: 8cm) = {
  figure(
    image("../img/" + source, width:width),
    caption: caption,

  )
}

#let listing(source, lang, caption: none) = {
  show raw: content => {
    set text(size: 10pt, font: "Cousine")
    set par(leading: 0.5em)
    content
  }
  let file = read("../listing/" + source)
  let code = raw(file, lang:lang)
  set par(first-line-indent: 0pt)
  code
}

#let tbl(content, caption: none) = {
	show figure.caption: set align(left)

  figure(
    content,
    caption: figure.caption(
      caption,
      position: top
    ),
    supplement: "Таблица",
    kind: "table"
  )
}
