[dificil]

A semântica do Python difere da do Lox em vários detalhes. Em muitos lugares,
tomamos atalhos e usamos o comportamento do Python como referência ao invés de
se basear no que a linguagem Lox deveria fazer. 

Nesse exercício, vamos implementar algumas funções no módulo lox.runtime com o 
comportamento esperado em Lox ao invés de usar as implementações padrão, como 
estamos fazendo.


## Imprimindo valores

Para vários valores, str(x) se comporta de forma diferente que o comportamento
esperado em Lox. Defina a função lox.runtime.show(x) que recebe um objeto e 
retorna uma string com a representação do mesmo em Lox (não em Python).
Implemente os casos especiais da função show:

- show(None) -> `nil`
- show(True) -> `true`
- show(False) -> `false`
- show(42.0) -> `42`
- show(SomeClass) -> `SomeClass`
- show(lox_instance) -> `SomeClass instance`
- show(lox_func) -> `<fn lox_func>`
- show(py_func) -> `<native fn>`

Você deve trocar a chamada de print(valor) na implementação de
`lox.ast.Print.eval` para usar print(show(valor)), convertendo o objeto Python
na representação desejada em Lox.

DICA: talvez você precise importar os tipos FunctionTypes e BuiltinFunctionType
do módulo types para verificar os objetos do tipo função nativa.

DICA: considere implementar o método __str__ de LoxFunction, LoxClass e
LoxInstance para controlar a conversão para strings e também simplificar a
implementação da função show().

DICA: str.removesuffix() pode facilitar a implementação de show para números.


## Operações matemáticas

Estamos usando, por enquanto, as funções matemáticas do modulo operator. Mas existem
diferenças entre os operadores Python e Lox e portanto devemos reimplementá-las
de acordo com o comportamento esperado em Lox:

- sub, mul, truediv: python aceita operações matemáticas com booleanos, 
  em Lox somente números podem participar de operações matemáticas. Levante um 
  LoxError ou TypeError quando necessário.
- add: como acima, mas também aceita soma de strings no Lox, assim como em Python.
- ge, le, gt, lt: também deveriam levantar erros em comparações que não envolvem números.


## Comparações

- eq, ne: Lox sempre trata valores de tipos diferentes como sendo diferentes. Python
  aceita comparações de números com booleanos como em 1.0 == True.
- truthy: em Lox, somente true e nil são falsos. 0 é considerado verdadeiro, ao contrário de Python.
- not_: deve obedecer à semantica de truthy e não de Python.

