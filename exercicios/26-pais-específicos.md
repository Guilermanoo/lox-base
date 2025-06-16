[medio]

Os comandos "this", "super" e "return" só podem aparecer dentro dentro de
funções, sendo que os segundos devem estar dentro de funções declaradas como
métodos de uma classe.

Para determinarmos se essas expressões estão declaradas nos lugares corretos, é
necessário portanto iterar sobre todos os pais das mesmas. Isso parece
complicado já que nossas classes só guardam referências sobre os filhos e demais
descendentes.

Por causa dessa limitação, o método `Node.validate_self(cursor)` recebe um
cursor como argumento. Cursores são objetos usados para navegar em estruturas de
dados e, no caso das árvores sintáticas do Lox, guardam um referências
explicitas aos pais e todos os nós ancestrais.


`Node.validate_self(cursor)` recebe um cursor apontando para o nó atual. O nó
para o qual o cursor aponta fica guardado no atributo `cursor.node`. O que é mais
relevante para as nossas necessidades é que cursores possuem alguns métodos 
importantes para navegar nas árvores:

* Cursor.children()
    - Itera sobre os filhos diretos do nó atual.
* Cursor.descendents()
    - Itera sobre os descendentes (filhos, filhos de filhos e assim por diante). 
* Cursor.siblings()
    - Itera sobre os irmãos (filhos do mesmo pai). 
* Cursor.parent() 
    - Retorna uma referência para o nó pai.
* Cursor.parents()
    - Itera sobre todos antecessores (pai, pai do pai e assim por diante).

Note que esses métodos sempre retoram cursores. Portanto lembre de acessar o 
atributo `.node` quando precisar de uma referência a um nó da árvore sintática.

```python
cursor.parent()       # cursor apontando para o pai
cursor.parent().node  # nó pai
```

Agora que sabemos acessar os pais de um nó, é possível fazer as verificações
adicionais mencionadas no início da atividade. Implemente o método
`.validate_self(cursor)` das classes `Super`, `This` e `Return` para garantir que 
as mesmas só ocorrem em lugares válidos:

* Return -> deve ser descendente de algum nó do tipo `Function`
* This -> deve ser descendente de algum nó do tipo `Class`
* Super -> deve ser descendente de algum nó di tipo `Class`. Essa classe deve
  herdar de alguma outra classe.
