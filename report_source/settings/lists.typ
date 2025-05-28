
#let lists(doc) = {
  // Добавляем отступы для первого уровня списка
  set list(indent: 1.25cm, marker: ([•], [•], [•]))
  show list: it => {
    set list(indent: 0cm)
    set enum(indent: 0cm)
    it
  }

  // Добавляем отступы для первого уровня списка
  set enum(indent: 1.25cm)
  show enum: it => {
    set enum(indent: 0cm)
    set list(indent: 0cm)
    it
  }

  doc
}
