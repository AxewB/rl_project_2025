#let appendix(doc) = {
  set heading(numbering: "A" )
  counter(heading).update(0)

  show heading: h => {
    set align(center)
    set text(size: 14pt)
    let number = counter(heading).display("A")

    [ПРИЛОЖЕНИЕ #number]
  }

  doc
}

#let apdx(caption) = {
  pagebreak()
  set align(center)
  show text: txt => strong(txt)

  let headingNumber = context(counter(heading).display())

  heading([ПРИЛОЖЕНИЕ #headingNumber], numbering: "A")
  text(caption)
}
