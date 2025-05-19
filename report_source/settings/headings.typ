#let headingNumberingFunction = (..nums) => {
  let result = nums.pos().map(str).slice(1).join(".")

  if (nums.pos().map(str).slice(1).len() > 0) {
    result + "."
  }
  else {
    result
  }
}

#let headings(doc) = {
  show heading.where(level: 1): set heading(numbering: headingNumberingFunction)
  show heading.where(level: 2): set heading(numbering: headingNumberingFunction)
  show heading.where(level: 3): set heading(numbering: headingNumberingFunction)
  show heading.where(level: 4): set heading(numbering: headingNumberingFunction)
  show heading: it => {
    set text(size: 14pt)
    if (it.level == 1) {
      set align(center)
      show text: t => upper(t)

      pad(left: 0cm, bottom: 0.5em, top: 0.5em, it)
    } else {
      pad(left: 1.25cm, bottom: 0.5em, top: 0.5em, it)
    }
  }

  // show heading: it => {
  //   set text(size: 14pt)
  //   set align(left)
  //   pad(left: 1.25cm, bottom: 0.5em, it)
  // }

  doc
}
